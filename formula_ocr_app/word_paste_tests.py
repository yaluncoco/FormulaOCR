from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import tkinter as tk

from formula_ocr_app.formula_formats import latex_to_mathml
from formula_ocr_app.word_clipboard import (
    FORMAT_HTML,
    FORMAT_MATHML,
    FORMAT_MATHML_PRESENTATION,
    FORMAT_OFFICE_OPEN_XML,
    copy_mathml_for_word_to_clipboard,
    windows_clipboard_formats,
    windows_clipboard_text,
)


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def run_word_paste_smoke(latex: str, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_path = output_dir / "word_paste_smoke.docx"
    source_path = output_dir / "word_paste_source.json"

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        mathml = latex_to_mathml(latex)
        copied = copy_mathml_for_word_to_clipboard(
            mathml,
            plain_text=latex,
            clipboard_widget=root,
            owner_hwnd=root.winfo_id(),
        )
        formats = windows_clipboard_formats()
        legacy_formats = {
            FORMAT_OFFICE_OPEN_XML,
            FORMAT_MATHML,
            FORMAT_MATHML_PRESENTATION,
        }
        source = {
            "latex": latex,
            "copied": copied,
            "formats": formats,
            "legacy_formats_absent": legacy_formats.isdisjoint(formats),
            "office_open_xml": (
                windows_clipboard_text(FORMAT_OFFICE_OPEN_XML)
                if FORMAT_OFFICE_OPEN_XML in formats
                else ""
            ),
            "mathml": (
                windows_clipboard_text(FORMAT_MATHML)
                if FORMAT_MATHML in formats
                else ""
            ),
            "mathml_presentation": (
                windows_clipboard_text(FORMAT_MATHML_PRESENTATION)
                if FORMAT_MATHML_PRESENTATION in formats
                else ""
            ),
            "html": (
                windows_clipboard_text(FORMAT_HTML) if FORMAT_HTML in formats else ""
            ),
        }
        source_path.write_text(
            json.dumps(source, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        root.destroy()

    if not copied:
        return {
            "copied": False,
            "formats": formats if "formats" in locals() else [],
            "error": "native Word clipboard write failed",
        }

    paste_result = _paste_clipboard_into_word(docx_path)
    analysis = analyze_docx_math(docx_path)
    return {
        "copied": True,
        "formats": formats,
        "docx": str(docx_path),
        "source": str(source_path),
        "paste": paste_result,
        "analysis": analysis,
    }


def _paste_clipboard_into_word(docx_path: Path) -> dict[str, object]:
    script_path = docx_path.with_suffix(".paste.ps1")
    script_path.write_text(
        """
$ErrorActionPreference = "Stop"
$docxPath = $args[0]
$word = $null
$doc = $null
$createdWord = $false
$previousVisible = $null
$previousAlerts = $null
try {
  try {
    $word = [Runtime.InteropServices.Marshal]::GetActiveObject("Word.Application")
  } catch {
    $word = New-Object -ComObject Word.Application
    $createdWord = $true
  }
  $previousVisible = $word.Visible
  $previousAlerts = $word.DisplayAlerts
  if ($createdWord) {
    $word.Visible = $false
  }
  $word.DisplayAlerts = 0
  $doc = $word.Documents.Add()
  $doc.Activate()
  $word.Selection.Paste()
  Start-Sleep -Milliseconds 800
  $omathCount = $doc.OMaths.Count
  $doc.SaveAs2($docxPath, 16)
  Write-Output ("omath_count=" + $omathCount)
} finally {
  if ($null -ne $doc) {
    $doc.Close($false) | Out-Null
  }
  if ($null -ne $word) {
    if ($null -ne $previousAlerts) {
      $word.DisplayAlerts = $previousAlerts
    }
    if ($createdWord) {
      $word.Quit() | Out-Null
    } elseif ($null -ne $previousVisible) {
      $word.Visible = $previousVisible
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            str(docx_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Word paste automation failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    omath_count = 0
    for line in completed.stdout.splitlines():
        if line.startswith("omath_count="):
            omath_count = int(line.partition("=")[2])
    return {"omath_count": omath_count, "stdout": completed.stdout.strip()}


def analyze_docx_math(docx_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(docx_path) as docx:
        document_xml = docx.read("word/document.xml").decode("utf-8", "replace")
    document_xml_path = docx_path.with_suffix(".document.xml")
    document_xml_path.write_text(document_xml, encoding="utf-8")
    root = ElementTree.fromstring(document_xml)
    ns = {"m": MATH_NS, "w": WORD_NS}
    nary = root.findall(".//m:nary", ns)
    s_sub_sup = root.findall(".//m:sSubSup", ns)
    s_sub = root.findall(".//m:sSub", ns)
    s_sup = root.findall(".//m:sSup", ns)
    sums = root.findall('.//m:naryPr/m:chr[@m:val="∑"]', ns)
    lim_locs = [
        node.attrib.get(f"{{{MATH_NS}}}val", "")
        for node in root.findall(".//m:naryPr/m:limLoc", ns)
    ]
    return {
        "document_xml": str(document_xml_path),
        "nary_count": len(nary),
        "sum_nary_count": len(sums),
        "sSubSup_count": len(s_sub_sup),
        "sSub_count": len(s_sub),
        "sSup_count": len(s_sup),
        "limLoc_values": lim_locs,
    }


def main() -> None:
    if sys.platform != "win32":
        print("word-paste-smoke: skipped on non-Windows")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formula",
        default=(
            r"Cov(X,Y)=\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}"
            r"(x_i-\bar{x})(y_j-\bar{y})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "_word_paste_smoke"),
    )
    args = parser.parse_args()
    result = run_word_paste_smoke(args.formula, Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
