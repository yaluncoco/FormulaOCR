from __future__ import annotations

import re
import zipfile
from html import escape
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

try:  # Primary, high-fidelity LaTeX -> MathML conversion.
    from latex2mathml.converter import convert as _latex2mathml_convert
except Exception:  # pragma: no cover - falls back to the built-in parser.
    _latex2mathml_convert = None


MATHML_NS = "http://www.w3.org/1998/Math/MathML"
ElementTree.register_namespace("", MATHML_NS)


GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "phi": "φ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Omega": "Ω",
}

SYMBOLS = {
    "infty": "∞",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "times": "×",
    "cdot": "·",
    "ast": "∗",
    "pm": "±",
    "mp": "∓",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "int": "∫",
    "partial": "∂",
    "nabla": "∇",
    "approx": "≈",
    "equiv": "≡",
    "sim": "∼",
    "propto": "∝",
    "parallel": "∥",
    "nparallel": "∦",
    "mid": "∣",
    "vert": "|",
    "Vert": "∥",
    "lvert": "|",
    "rvert": "|",
    "lVert": "∥",
    "rVert": "∥",
    "leftarrow": "←",
    "rightarrow": "→",
    "to": "→",
    "Leftarrow": "⇐",
    "Rightarrow": "⇒",
    "leftrightarrow": "↔",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "∪",
    "cap": "∩",
    "forall": "∀",
    "exists": "∃",
}

DOUBLE_STRUCK = {
    "A": "𝔸",
    "B": "𝔹",
    "C": "ℂ",
    "D": "𝔻",
    "E": "𝔼",
    "F": "𝔽",
    "G": "𝔾",
    "H": "ℍ",
    "I": "𝕀",
    "J": "𝕁",
    "K": "𝕂",
    "L": "𝕃",
    "M": "𝕄",
    "N": "ℕ",
    "O": "𝕆",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "S": "𝕊",
    "T": "𝕋",
    "U": "𝕌",
    "V": "𝕍",
    "W": "𝕎",
    "X": "𝕏",
    "Y": "𝕐",
    "Z": "ℤ",
}

LARGE_OPERATORS = {"∑", "∏", "∐", "⋃", "⋂"}
WORD_LARGE_OPERATORS = {
    "∑",
    "∏",
    "∐",
    "⋃",
    "⋂",
    "∫",
    "∬",
    "∭",
    "∮",
    "∯",
    "∰",
}
WORD_UNDER_OVER_OPERATORS = {"∑", "∏", "∐", "⋃", "⋂"}
WORD_DELIMITER_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "|": "|",
    "‖": "‖",
    "⌊": "⌋",
    "⌈": "⌉",
    "⟨": "⟩",
    "〈": "〉",
}
WORD_DELIMITER_OPERATORS = set(WORD_DELIMITER_PAIRS) | set(
    WORD_DELIMITER_PAIRS.values()
)


def normalize_latex(latex: str) -> str:
    text = latex.strip()
    text = re.sub(r"^\$\$?|\$\$?$", "", text).strip()
    text = re.sub(r"^\\\[|\\\]$", "", text).strip()
    text = re.sub(r"^\\\(|\\\)$", "", text).strip()
    return text


def clean_recognized_latex(latex: str) -> str:
    text = normalize_latex(latex)
    text = _normalize_ocr_separator_punctuation(text)
    commands: list[str] = []

    def _save_command(match: re.Match[str]) -> str:
        commands.append(match.group(0))
        return f"\uFFF0{len(commands) - 1}\uFFF1"

    protected = re.sub(r"\\[A-Za-z]+|\\.", _save_command, text)
    previous = None
    while previous != protected:
        previous = protected
        protected = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1\2", protected)
    protected = re.sub(r"([A-Za-z0-9}])\s+([_^])", r"\1\2", protected)
    protected = re.sub(r"([_^])\s+([A-Za-z0-9{])", r"\1\2", protected)

    def _restore_command(match: re.Match[str]) -> str:
        return commands[int(match.group(1))]

    text = re.sub(r"\uFFF0(\d+)\uFFF1", _restore_command, protected)
    text = re.sub(
        r"\\(mathbb|mathcal|mathbf|mathit|mathsf|mathtt|mathfrak)\s+([A-Za-z])",
        r"\\\1{\2}",
        text,
    )
    text = _normalize_named_operators(text)
    text = _normalize_ocr_relation_artifacts(text)
    text = _normalize_named_operator_limits(text)
    return text.strip()


def _normalize_ocr_separator_punctuation(text: str) -> str:
    text = text.replace("，", ",")
    # Formula OCR often mistakes an argument-separating comma after a subscript
    # group for a prime-like quote. Keep real primes such as f'(x) untouched.
    return re.sub(r"(?<=[}\]])\s*['’‘′‵ʹ]\s*(?=(?:[A-Za-z]|\\[A-Za-z]+))", ",", text)


_NAMED_OPERATOR_COMMANDS = {
    "arg": r"\arg",
    "max": r"\max",
    "min": r"\min",
    "sup": r"\sup",
    "inf": r"\inf",
    "lim": r"\lim",
    "det": r"\det",
    "gcd": r"\gcd",
    "log": r"\log",
    "ln": r"\ln",
    "exp": r"\exp",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
}


def _normalize_named_operators(text: str) -> str:
    """Turn verbose OCR operator markup into stable standard commands."""

    def _replace(match: re.Match[str]) -> str:
        name = re.sub(r"\s+", "", match.group(1)).lower()
        if name in {"argmax", "argmin"}:
            return "\\arg\\" + name[3:]
        return _NAMED_OPERATOR_COMMANDS.get(name, match.group(0))

    return re.sub(
        r"\\operatorname\*?\s*\{\s*([A-Za-z][A-Za-z\s]*?)\s*\}",
        _replace,
        text,
    )


def _normalize_ocr_relation_artifacts(text: str) -> str:
    """Repair high-confidence spacing/equality artifacts emitted by OCR."""

    # A spacing command immediately before an equality sign is visual noise.
    text = re.sub(r"(?:\\[ ,;:!]\s*)+(?=\\?=)", "", text)
    # Formula images contain mathematical equality rather than programming
    # comparisons.  Paddle occasionally emits ``\==`` or ``==`` for one glyph.
    return re.sub(r"\\?={2,}", "=", text)


def _normalize_named_operator_limits(text: str) -> str:
    r"""Preserve a visibly under-set optimizer variable from formula images.

    Paddle's formula models commonly emit ``\arg\max_{k}`` for a display
    formula whose ``k`` is centered below ``max``.  The spelling is valid
    LaTeX, but several downstream converters treat it as an ordinary right
    subscript.  Adding the explicit ``\limits`` command retains the original
    two-dimensional layout in the editable result, MathML preview and Word.
    """

    return re.sub(
        r"(\\arg\s*\\(?:max|min))(?!\s*\\(?:limits|nolimits))(\s*_)",
        r"\1\\limits\2",
        text,
    )


_EMPTY_MATHML = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
    "<mrow></mrow></math>"
)

# latex2mathml (as of 3.81) emits malformed XML for a handful of multi-line
# environments. Remap them to equivalents it renders correctly before parsing.
_ENV_FALLBACK = {
    "aligned": "align",
    "gathered": "align",
    "gather": "align",
}


def _prepare_latex2mathml_input(text: str) -> str:
    for bad, good in _ENV_FALLBACK.items():
        text = text.replace(f"\\begin{{{bad}}}", f"\\begin{{{good}}}")
        text = text.replace(f"\\end{{{bad}}}", f"\\end{{{good}}}")
    # latex2mathml 3.81 renders the standard ``\arg`` command as literal text
    # instead of a named operator.  Keep canonical LaTeX in the UI, but feed an
    # equivalent spelling to the converter for correct preview/Word output.
    text = re.sub(
        r"\\arg(?![A-Za-z])",
        lambda _match: r"\operatorname{arg}",
        text,
    )
    return text


def latex_to_mathml(latex: str) -> str:
    """Convert LaTeX to presentation MathML.

    Uses the ``latex2mathml`` library for high-fidelity output (roots, accents,
    matrices, binomials, piecewise functions, etc.). Produces clean
    presentation MathML without a ``<semantics>``/``<annotation>`` wrapper,
    which is the most widely compatible form for pasting into Word/WPS equation
    editors. Falls back to the built-in parser if the library is unavailable or
    fails on a given input.
    """
    normalized = normalize_latex(latex)
    if not normalized:
        return _EMPTY_MATHML
    if _latex2mathml_convert is not None:
        try:
            prepared = _prepare_latex2mathml_input(normalized)
            mathml = _latex2mathml_convert(prepared, display="block")
            # Guard against any malformed output before handing it downstream.
            ElementTree.fromstring(mathml)
            return mathml
        except Exception:
            pass
    return _latex_to_mathml_legacy(latex)


def latex_to_markdown_inline(latex: str) -> str:
    return f"${normalize_latex(latex)}$"


def latex_to_markdown_block(latex: str) -> str:
    return f"$$\n{normalize_latex(latex)}\n$$"


def latex_to_equation_environment(latex: str) -> str:
    return f"\\begin{{equation}}\n{normalize_latex(latex)}\n\\end{{equation}}"


def latex_to_html(latex: str) -> str:
    mathml = latex_to_mathml(latex)
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8"></head>'
        f"<body>{mathml}</body></html>"
    )


def mathml_to_omml(mathml: str) -> str:
    """Convert presentation MathML to the Office Math Markup Word expects."""
    root = ElementTree.fromstring(mathml)
    body = _omml_children(root)
    return (
        '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
        f"<m:oMath>{body}</m:oMath>"
        "</m:oMathPara>"
    )


def mathml_to_word_mathml(mathml: str) -> str:
    """Normalize MathML to the Presentation MathML flavor Word places on copy."""
    root = ElementTree.fromstring(mathml)
    _normalize_word_mathml_tree(root)
    return (
        '<?xml version="1.0"?>\r\n'
        + ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)
        + "\r\n"
    )


def _normalize_word_mathml_tree(element: ElementTree.Element) -> None:
    _normalize_word_large_operator_limits(element)
    tag = _local_name(element.tag)
    if tag == "math":
        element.attrib["display"] = "block"
    if tag == "mo" and element.attrib.get("stretchy") == "false":
        element.attrib.pop("stretchy", None)
    if tag == "mo" and (
        _element_text(element).strip() in WORD_DELIMITER_OPERATORS
        or element.attrib.get("fence") == "true"
    ):
        _normalize_word_fence_operator(element)
    if tag == "mo" and _element_text(element).strip() in {",", "，"}:
        _normalize_word_comma_operator(element)
    if tag == "mo" and _element_text(element).strip() in WORD_LARGE_OPERATORS:
        element.attrib["largeop"] = "true"
        element.attrib["movablelimits"] = "false"
    if tag in {"munder", "mover", "munderover"}:
        _wrap_word_large_operator_limit_base(element)
    if tag == "mover":
        _normalize_word_overline(element)
    for child in list(element):
        _normalize_word_mathml_tree(child)
    _flatten_word_argument_list_groups(element)
    _insert_word_large_operator_spacing(element)


def _normalize_word_comma_operator(element: ElementTree.Element) -> None:
    element.text = ","
    element.attrib.pop("separator", None)
    element.attrib.pop("lspace", None)
    element.attrib.pop("rspace", None)


def _normalize_word_fence_operator(element: ElementTree.Element) -> None:
    element.attrib.pop("fence", None)
    element.attrib.pop("form", None)
    element.attrib.pop("stretchy", None)


def _normalize_word_large_operator_limits(element: ElementTree.Element) -> None:
    tag = _local_name(element.tag)
    if tag not in {"msub", "msup", "msubsup"}:
        return
    children = list(element)
    if not children or not _is_word_under_over_operator_element(children[0]):
        return
    if tag == "msubsup" and len(children) >= 3:
        element.tag = _mathml_tag_like(element, "munderover")
    elif tag == "msub" and len(children) >= 2:
        element.tag = _mathml_tag_like(element, "munder")
    elif tag == "msup" and len(children) >= 2:
        element.tag = _mathml_tag_like(element, "mover")


def _wrap_word_large_operator_limit_base(element: ElementTree.Element) -> None:
    children = list(element)
    if not children:
        return
    base = children[0]
    if _local_name(base.tag) == "mrow":
        return
    if not _is_word_under_over_operator_element(base):
        return

    base_tail = base.tail
    base.tail = None
    element.remove(base)
    wrapper = ElementTree.Element(_mathml_tag_like(element, "mrow"))
    wrapper.append(base)
    wrapper.tail = base_tail
    element.insert(0, wrapper)


def _insert_word_large_operator_spacing(element: ElementTree.Element) -> None:
    if _local_name(element.tag) not in {"math", "mrow", "mtd", "mstyle"}:
        return
    index = 0
    while index < len(element) - 1:
        child = element[index]
        next_child = element[index + 1]
        if _is_word_large_operator_limit_element(child) and not _is_word_spacing_element(
            next_child
        ):
            spacer = ElementTree.Element(_mathml_tag_like(element, "mrow"))
            mo = ElementTree.SubElement(spacer, _mathml_tag_like(element, "mo"))
            mo.text = "\u200a"
            spacer.tail = child.tail
            child.tail = None
            element.insert(index + 1, spacer)
            index += 2
            continue
        index += 1


def _flatten_word_argument_list_groups(element: ElementTree.Element) -> None:
    if _local_name(element.tag) not in {"math", "mrow", "mtd", "mstyle"}:
        return
    index = 0
    while index < len(element):
        child = element[index]
        if not _is_word_argument_list_group(child):
            index += 1
            continue

        children = list(child)
        group_tail = child.tail
        child.tail = None
        element.remove(child)
        for offset, nested in enumerate(children):
            child.remove(nested)
            element.insert(index + offset, nested)
        if children:
            children[-1].tail = (children[-1].tail or "") + (group_tail or "")
            index += len(children)
        else:
            index += 1


def _is_word_argument_list_group(element: ElementTree.Element) -> bool:
    if _local_name(element.tag) != "mrow":
        return False
    children = list(element)
    if len(children) < 3:
        return False
    opening = _element_text(children[0]).strip()
    closing = _element_text(children[-1]).strip()
    return (
        WORD_DELIMITER_PAIRS.get(opening) == closing
        and any(_is_word_comma_element(child) for child in children[1:-1])
    )


def _is_word_operator_text(element: ElementTree.Element, text: str) -> bool:
    return _local_name(element.tag) == "mo" and _element_text(element).strip() == text


def _is_word_comma_element(element: ElementTree.Element) -> bool:
    return _local_name(element.tag) == "mo" and _element_text(element).strip() == ","


def _is_word_large_operator_limit_element(element: ElementTree.Element) -> bool:
    if _local_name(element.tag) not in {"munder", "mover", "munderover"}:
        return False
    children = list(element)
    return bool(children) and _is_word_under_over_operator_element(children[0])


def _is_word_spacing_element(element: ElementTree.Element) -> bool:
    if _local_name(element.tag) == "mspace":
        return True
    if _local_name(element.tag) == "mo":
        return _element_text(element) in {"\u200a", "\u2009", "\u2062"}
    children = list(element)
    return (
        _local_name(element.tag) == "mrow"
        and len(children) == 1
        and _is_word_spacing_element(children[0])
    )


def _is_word_under_over_operator_element(element: ElementTree.Element) -> bool:
    return _element_text(element).strip() in WORD_UNDER_OVER_OPERATORS


def _mathml_tag_like(element: ElementTree.Element, local_name: str) -> str:
    if "}" not in element.tag:
        return local_name
    namespace = element.tag.split("}", 1)[0] + "}"
    return f"{namespace}{local_name}"


def _is_large_operator_element(element: ElementTree.Element) -> bool:
    return (
        _local_name(element.tag) == "mo"
        and _element_text(element).strip() in WORD_LARGE_OPERATORS
    )


def _normalize_word_overline(element: ElementTree.Element) -> None:
    children = list(element)
    if len(children) < 2:
        return
    accent = _element_text(children[1]).strip()
    if accent not in {"¯", "―", "‾", "\u0305"}:
        return
    element.attrib.clear()
    element.attrib["accent"] = "true"

    base = children[0]
    if _local_name(base.tag) != "mrow":
        base_tail = base.tail
        base.tail = None
        element.remove(base)
        wrapper = ElementTree.Element(f"{{{MATHML_NS}}}mrow")
        wrapper.append(base)
        wrapper.tail = base_tail
        element.insert(0, wrapper)

    accent_node = children[1]
    accent_tail = accent_node.tail
    element.remove(accent_node)
    replacement = ElementTree.Element(f"{{{MATHML_NS}}}mi")
    replacement.text = "―"
    replacement.tail = accent_tail
    element.insert(1, replacement)


def _omml_children(element: ElementTree.Element) -> str:
    return "".join(_omml_element(child) for child in list(element))


def _omml_element(element: ElementTree.Element) -> str:
    tag = _local_name(element.tag)
    if tag in {"math", "mrow", "mstyle", "semantics", "mtd"}:
        return _omml_text(element.text) + _omml_children(element) + _omml_text(element.tail)
    if tag == "annotation":
        return _omml_text(element.tail)
    if tag in {"mi", "mn", "mo", "mtext"}:
        return _omml_text(element.text or "") + _omml_text(element.tail)
    if tag == "mspace":
        if (element.attrib.get("linebreak") or "").lower() == "newline":
            return '<m:r><m:br/></m:r>' + _omml_text(element.tail)
        return _omml_text(" ") + _omml_text(element.tail)
    if tag == "mfrac":
        children = list(element)
        numerator = _omml_element(children[0]) if children else ""
        denominator = _omml_element(children[1]) if len(children) > 1 else ""
        return (
            "<m:f><m:num>"
            f"{numerator}"
            "</m:num><m:den>"
            f"{denominator}"
            "</m:den></m:f>"
            + _omml_text(element.tail)
        )
    if tag == "msqrt":
        return (
            '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>'
            f"{_omml_children(element)}"
            "</m:e></m:rad>"
            + _omml_text(element.tail)
        )
    if tag == "msup":
        children = list(element)
        if children and _is_large_operator_element(children[0]):
            return _omml_nary(
                children[0],
                None,
                children[1] if len(children) > 1 else None,
                tail=element.tail,
            )
        base = _omml_element(children[0]) if children else ""
        sup = _omml_element(children[1]) if len(children) > 1 else ""
        return (
            f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
            + _omml_text(element.tail)
        )
    if tag == "msub":
        children = list(element)
        if children and _is_large_operator_element(children[0]):
            return _omml_nary(
                children[0],
                children[1] if len(children) > 1 else None,
                None,
                tail=element.tail,
            )
        base = _omml_element(children[0]) if children else ""
        sub = _omml_element(children[1]) if len(children) > 1 else ""
        return (
            f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
            + _omml_text(element.tail)
        )
    if tag == "msubsup":
        children = list(element)
        if children and _is_large_operator_element(children[0]):
            return _omml_nary(
                children[0],
                children[1] if len(children) > 1 else None,
                children[2] if len(children) > 2 else None,
                tail=element.tail,
            )
        base = _omml_element(children[0]) if children else ""
        sub = _omml_element(children[1]) if len(children) > 1 else ""
        sup = _omml_element(children[2]) if len(children) > 2 else ""
        return (
            f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
            + _omml_text(element.tail)
        )
    if tag == "munderover":
        children = list(element)
        if children and _is_large_operator_element(children[0]):
            return _omml_nary(
                children[0],
                children[1] if len(children) > 1 else None,
                children[2] if len(children) > 2 else None,
                tail=element.tail,
            )
        base = _omml_element(children[0]) if children else ""
        sub = _omml_element(children[1]) if len(children) > 1 else ""
        sup = _omml_element(children[2]) if len(children) > 2 else ""
        return (
            f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
            + _omml_text(element.tail)
        )
    if tag == "munder":
        children = list(element)
        if children and _is_large_operator_element(children[0]):
            return _omml_nary(
                children[0],
                children[1] if len(children) > 1 else None,
                None,
                tail=element.tail,
            )
        base = _omml_element(children[0]) if children else ""
        sub = _omml_element(children[1]) if len(children) > 1 else ""
        return (
            f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
            + _omml_text(element.tail)
        )
    if tag == "mover":
        children = list(element)
        base = _omml_element(children[0]) if children else ""
        if children and _is_large_operator_element(children[0]):
            return _omml_nary(
                children[0],
                None,
                children[1] if len(children) > 1 else None,
                tail=element.tail,
            )
        accent = _element_text(children[1]) if len(children) > 1 else "¯"
        if not accent or len(accent.strip()) > 2:
            accent = "¯"
        return (
            f'<m:acc><m:accPr><m:chr m:val="{escape(accent.strip())}"/></m:accPr>'
            f"<m:e>{base}</m:e></m:acc>"
            + _omml_text(element.tail)
        )
    if tag == "mtable":
        rows = []
        for row in list(element):
            cells = "".join(f"<m:e>{_omml_element(cell)}</m:e>" for cell in list(row))
            rows.append(f"<m:mr>{cells}</m:mr>")
        return f"<m:m>{''.join(rows)}</m:m>" + _omml_text(element.tail)
    if tag == "mtr":
        return _omml_children(element) + _omml_text(element.tail)
    return _omml_text(element.text) + _omml_children(element) + _omml_text(element.tail)


def _omml_nary(
    operator: ElementTree.Element,
    subscript: ElementTree.Element | None,
    superscript: ElementTree.Element | None,
    *,
    tail: str | None,
) -> str:
    symbol = escape(_element_text(operator).strip() or "∑")
    lim_loc = (
        "undOvr"
        if _element_text(operator).strip() in WORD_UNDER_OVER_OPERATORS
        else "subSup"
    )
    sub = _omml_element(subscript) if subscript is not None else ""
    sup = _omml_element(superscript) if superscript is not None else ""
    return (
        "<m:nary>"
        f'<m:naryPr><m:chr m:val="{symbol}"/>'
        f'<m:limLoc m:val="{lim_loc}"/><m:grow m:val="1"/></m:naryPr>'
        f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e></m:e>"
        "</m:nary>"
        + _omml_text(tail)
    )


def _omml_text(text: str | None) -> str:
    if not text:
        return ""
    return f'<m:r><m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _element_text(element: ElementTree.Element) -> str:
    parts = [element.text or ""]
    for child in list(element):
        parts.append(_element_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _latex_to_mathml_legacy(latex: str) -> str:
    normalized = normalize_latex(latex)
    rows = _split_latex_rows(normalized)
    if len(rows) > 1:
        body = (
            '<mtable columnalign="left">'
            + "".join(
                f"<mtr><mtd>{_parse_mathml_fragment(row)}</mtd></mtr>"
                for row in rows
            )
            + "</mtable>"
        )
    else:
        body = _parse_mathml_fragment(normalized)
    original = escape(normalize_latex(latex))
    return (
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
        f"<semantics>{body}<annotation encoding=\"application/x-tex\">{original}</annotation>"
        "</semantics></math>"
    )


def latex_to_asciimath(latex: str) -> str:
    text = _plain_formula(normalize_latex(latex))
    text = _convert_two_arg_command(text, "frac", lambda a, b: f"(({a})/({b}))")
    text = _convert_one_arg_command(text, "sqrt", lambda a: f"sqrt({a})")
    text = _convert_one_arg_command(text, "mathcal", lambda a: a)
    text = _convert_one_arg_command(text, "mathrm", lambda a: a)
    text = _convert_one_arg_command(text, "mathbf", lambda a: a)
    text = _convert_scripts(text)
    for name, value in sorted({**GREEK, **SYMBOLS}.items(), key=lambda item: -len(item[0])):
        ascii_value = {
            "∞": "inf",
            "≤": "<=",
            "≥": ">=",
            "≠": "!=",
            "×": "*",
            "·": "*",
            "±": "+-",
            "∓": "-+",
            "∑": "sum",
            "∫": "int",
            "∂": "del",
            "∇": "grad",
            "≈": "~~",
        }.get(value, value)
        text = _replace_latex_command(text, name, ascii_value)
    return _collapse_spaces(_strip_remaining_latex(text))


def latex_to_typst(latex: str) -> str:
    text = _plain_formula(normalize_latex(latex))
    text = _convert_two_arg_command(text, "frac", lambda a, b: f"frac({a}, {b})")
    text = _convert_one_arg_command(text, "sqrt", lambda a: f"sqrt({a})")
    text = _convert_one_arg_command(text, "mathcal", lambda a: f"cal({a})")
    text = _convert_one_arg_command(text, "mathrm", lambda a: f'upright("{_strip_remaining_latex(a)}")')
    text = _convert_one_arg_command(text, "mathbf", lambda a: f"bold({a})")
    text = _convert_scripts(text)
    typst_symbols = {
        "infty": "infinity",
        "le": "<=",
        "leq": "<=",
        "ge": ">=",
        "geq": ">=",
        "neq": "!=",
        "times": "times",
        "cdot": "dot",
        "sum": "sum",
        "int": "integral",
        "partial": "diff",
        "approx": "approx",
    }
    for name, value in sorted(typst_symbols.items(), key=lambda item: -len(item[0])):
        text = _replace_latex_command(text, name, value)
    for name, value in sorted(GREEK.items(), key=lambda item: -len(item[0])):
        text = _replace_latex_command(text, name, value)
    return f"$ {_collapse_spaces(_strip_remaining_latex(text))} $"


def latex_to_word_linear(latex: str) -> str:
    text = _plain_formula(normalize_latex(latex)).replace("; ", "\n")
    text = _convert_overline_commands(text)
    text = _convert_two_arg_command(
        text,
        "frac",
        lambda a, b: f"({_word_part(a)})/({_word_part(b)})",
    )
    text = _convert_one_arg_command(text, "sqrt", lambda a: f"√({_word_part(a)})")
    text = _convert_one_arg_command(text, "mathcal", _word_part)
    text = _convert_one_arg_command(text, "mathrm", _word_part)
    text = _convert_one_arg_command(text, "mathbf", _word_part)
    text = _convert_one_arg_command(text, "operatorname", _word_part)
    text = _convert_one_arg_command(text, "text", _word_part)
    text = _convert_scripts(text)
    text = _replace_word_symbols(text)
    text = _strip_remaining_latex(text)
    lines = [_collapse_spaces(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _convert_overline_commands(text: str) -> str:
    for command in ("overline", "bar"):
        text = _convert_accent_command(text, command, _with_combining_overline)
    return text


def _convert_accent_command(text: str, command: str, formatter) -> str:
    pattern = f"\\{command}"
    index = 0
    while True:
        index = text.find(pattern, index)
        if index < 0:
            return text
        atom_start = index + len(pattern)
        while atom_start < len(text) and text[atom_start].isspace():
            atom_start += 1
        if atom_start >= len(text):
            index += len(pattern)
            continue
        if text[atom_start] == "{":
            atom_end = _matching_brace(text, atom_start)
            if atom_end < 0:
                index += len(pattern)
                continue
            inner = text[atom_start + 1 : atom_end]
            replacement = formatter(inner)
            text = text[:index] + replacement + text[atom_end + 1 :]
            index += len(replacement)
            continue
        if text[atom_start] == "\\":
            match = re.match(r"\\[A-Za-z]+|\\.", text[atom_start:])
            if not match:
                index += len(pattern)
                continue
            atom_end = atom_start + len(match.group(0))
            inner = match.group(0)
        else:
            atom_end = atom_start + 1
            inner = text[atom_start:atom_end]
        replacement = formatter(inner)
        text = text[:index] + replacement + text[atom_end:]
        index += len(replacement)


def _with_combining_overline(text: str) -> str:
    plain = _strip_remaining_latex(text)
    return "".join(
        f"{char}\u0305" if not char.isspace() else char for char in plain
    )


def export_formula_docx(
    output_path: str | Path,
    *,
    latex: str,
    mathml: str,
    asciimath: str,
    typst: str,
    word_linear: str | None = None,
    image_path: str | Path | None = None,
) -> None:
    output_path = Path(output_path)
    image_path = Path(image_path) if image_path else None
    word_linear = word_linear or latex_to_word_linear(latex)
    try:
        word_formula_xml = f"<w:p>{mathml_to_omml(mathml)}</w:p>"
    except Exception:
        word_formula_xml = (
            f"<w:p><m:oMathPara><m:oMath><m:r><m:t>{escape(word_linear)}</m:t></m:r>"
            "</m:oMath></m:oMathPara></w:p>"
        )
    image_xml = ""
    rel_image = ""
    content_image = ""
    media_name = "image1.png"

    if image_path and image_path.exists():
        width_emu, height_emu = _image_extent_emu(image_path)
        image_xml = f"""
        <w:p>
          <w:r>
            <w:drawing>
              <wp:inline distT="0" distB="0" distL="0" distR="0">
                <wp:extent cx="{width_emu}" cy="{height_emu}"/>
                <wp:docPr id="1" name="Formula Image"/>
                <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                    <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                      <pic:nvPicPr><pic:cNvPr id="0" name="{media_name}"/><pic:cNvPicPr/></pic:nvPicPr>
                      <pic:blipFill><a:blip r:embed="rIdImage1"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
                      <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm><a:prstGeom prst="rect"/></pic:spPr>
                    </pic:pic>
                  </a:graphicData>
                </a:graphic>
              </wp:inline>
            </w:drawing>
          </w:r>
        </w:p>
        """
        rel_image = '<Relationship Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>'
        content_image = '<Default Extension="png" ContentType="image/png"/>'

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
  <w:body>
    {_paragraph("FormulaOCR 识别结果", bold=True)}
    {image_xml}
    {_paragraph("Word 公式", bold=True)}
    {word_formula_xml}
    {_paragraph("Word 线性兜底", bold=True)}
    {_paragraph(word_linear)}
    {_paragraph("LaTeX", bold=True)}
    {_paragraph(latex)}
    {_paragraph("MathML", bold=True)}
    {_paragraph(mathml)}
    {_paragraph("AsciiMath", bold=True)}
    {_paragraph(asciimath)}
    {_paragraph("Typst", bold=True)}
    {_paragraph(typst)}
    <w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>
"""
    rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rel_image}
</Relationships>
"""
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {content_image}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", package_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/_rels/document.xml.rels", rels_xml)
        if image_path and image_path.exists():
            docx.write(image_path, f"word/media/{media_name}")


class _LatexMathMLParser:
    def __init__(self, text: str) -> None:
        self.text = _plain_formula(text)
        self.pos = 0

    def parse(self) -> str:
        return f"<mrow>{self._parse_until('')}</mrow>"

    def _parse_until(self, stop: str) -> str:
        parts: list[str] = []
        while self.pos < len(self.text):
            if stop and self.text.startswith(stop, self.pos):
                self.pos += len(stop)
                break
            char = self.text[self.pos]
            if char == "}":
                if stop == "}":
                    self.pos += 1
                break
            if char == "{":
                self.pos += 1
                parts.append(f"<mrow>{self._parse_until('}')}</mrow>")
                continue
            if char in "^_":
                marker = char
                self.pos += 1
                script = self._parse_script_atom()
                base = parts.pop() if parts else "<mrow/>"
                parts.append(_apply_script(base, marker, script))
                continue
            parts.append(self._parse_atom())
        return "".join(parts)

    def _parse_atom(self) -> str:
        if self.pos >= len(self.text):
            return ""
        char = self.text[self.pos]
        if char == "{":
            self.pos += 1
            return f"<mrow>{self._parse_until('}')}</mrow>"
        if char == "\\":
            return self._parse_command()
        self.pos += 1
        if char.isspace():
            return ""
        if char == "~":
            return "<mspace width=\"0.25em\"/>"
        if char.isalpha():
            return f"<mi>{escape(char)}</mi>"
        if char.isdigit() or char == ".":
            return f"<mn>{escape(char)}</mn>"
        return f"<mo>{escape(char)}</mo>"

    def _parse_command(self) -> str:
        self.pos += 1
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalpha():
            self.pos += 1
        name = self.text[start:self.pos]
        if not name and self.pos < len(self.text):
            symbol = self.text[self.pos]
            self.pos += 1
            return "<mspace width=\"0.25em\"/>" if symbol in " ,;:" else f"<mo>{escape(symbol)}</mo>"
        if name in {
            "left",
            "right",
            "big",
            "Big",
            "bigg",
            "Bigg",
            "bigl",
            "bigr",
            "Bigl",
            "Bigr",
            "biggl",
            "biggr",
            "Biggl",
            "Biggr",
        }:
            return ""
        if name in {"begin", "end"}:
            self._read_group_text()
            return ""
        if name == "frac":
            numerator = self._parse_atom()
            denominator = self._parse_atom()
            return f"<mfrac>{numerator}{denominator}</mfrac>"
        if name == "sqrt":
            return f"<msqrt>{self._parse_atom()}</msqrt>"
        if name == "mathcal":
            text = self._read_group_text()
            return f'<mi mathvariant="script">{escape(text)}</mi>'
        if name == "mathbb":
            text = self._read_group_text()
            return _double_struck_mathml(text)
        if name in {"mathbf", "boldsymbol"}:
            return f'<mstyle mathvariant="bold">{self._parse_atom()}</mstyle>'
        if name == "mathit":
            return f'<mstyle mathvariant="italic">{self._parse_atom()}</mstyle>'
        if name == "mathsf":
            return f'<mstyle mathvariant="sans-serif">{self._parse_atom()}</mstyle>'
        if name == "mathtt":
            return f'<mstyle mathvariant="monospace">{self._parse_atom()}</mstyle>'
        if name == "mathfrak":
            return f'<mstyle mathvariant="fraktur">{self._parse_atom()}</mstyle>'
        if name in {"mathrm", "operatorname", "text"}:
            return f"<mtext>{escape(self._read_group_text())}</mtext>"
        if name in GREEK:
            return f"<mi>{GREEK[name]}</mi>"
        if name in SYMBOLS:
            return f"<mo>{SYMBOLS[name]}</mo>"
        return f"<mi>{escape(name)}</mi>"

    def _parse_script_atom(self) -> str:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        if self.pos >= len(self.text) or self.text[self.pos] != "{":
            return self._parse_atom()

        raw = self._read_group_raw()
        if re.fullmatch(r"[A-Za-z0-9\s]+", raw or ""):
            raw = re.sub(r"\s+", "", raw)
        return _parse_mathml_fragment(raw)

    def _read_group_raw(self) -> str:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        if self.pos >= len(self.text) or self.text[self.pos] != "{":
            return ""
        self.pos += 1
        depth = 1
        start = self.pos
        while self.pos < len(self.text) and depth:
            if self.text[self.pos] == "{":
                depth += 1
            elif self.text[self.pos] == "}":
                depth -= 1
            self.pos += 1
        return self.text[start : self.pos - 1]

    def _read_group_text(self) -> str:
        raw = self._read_group_raw()
        return _strip_remaining_latex(raw)


def _double_struck_mathml(text: str) -> str:
    clean = re.sub(r"\s+", "", text)
    if not clean:
        return "<mrow/>"
    parts = []
    for char in clean:
        if char in DOUBLE_STRUCK:
            parts.append(f"<mi>{DOUBLE_STRUCK[char]}</mi>")
        elif char.isdigit():
            parts.append(f"<mn>{escape(char)}</mn>")
        elif char.isalpha():
            parts.append(f'<mi mathvariant="double-struck">{escape(char)}</mi>')
        else:
            parts.append(f"<mo>{escape(char)}</mo>")
    if len(parts) == 1:
        return parts[0]
    return f"<mrow>{''.join(parts)}</mrow>"


def _apply_script(base: str, marker: str, script: str) -> str:
    if _is_large_operator(base):
        if marker == "^":
            split_under = _split_script_element(base, "munder")
            if split_under:
                base_node, under = split_under
                return f"<munderover>{base_node}{under}{script}</munderover>"
            return f"<mover>{base}{script}</mover>"

        split_over = _split_script_element(base, "mover")
        if split_over:
            base_node, over = split_over
            return f"<munderover>{base_node}{script}{over}</munderover>"
        return f"<munder>{base}{script}</munder>"

    if marker == "^":
        split_sub = _split_script_element(base, "msub")
        if split_sub:
            base_node, subscript = split_sub
            return f"<msubsup>{base_node}{subscript}{script}</msubsup>"
        return f"<msup>{base}{script}</msup>"

    split_sup = _split_script_element(base, "msup")
    if split_sup:
        base_node, superscript = split_sup
        return f"<msubsup>{base_node}{script}{superscript}</msubsup>"
    return f"<msub>{base}{script}</msub>"


def _is_large_operator(xml: str) -> bool:
    split_under = _split_script_element(xml, "munder")
    if split_under:
        return _is_large_operator(split_under[0])
    split_over = _split_script_element(xml, "mover")
    if split_over:
        return _is_large_operator(split_over[0])
    try:
        element = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return False
    return element.tag == "mo" and (element.text or "") in LARGE_OPERATORS


def _split_script_element(xml: str, tag: str) -> tuple[str, str] | None:
    try:
        element = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return None
    if element.tag != tag:
        return None
    children = list(element)
    if len(children) != 2:
        return None
    return (
        ElementTree.tostring(children[0], encoding="unicode", short_empty_elements=False),
        ElementTree.tostring(children[1], encoding="unicode", short_empty_elements=False),
    )


def _plain_formula(text: str) -> str:
    text = text.replace("\\\\", "; ")
    text = re.sub(r"\\begin\{[^}]+\}(?:\{[^}]*\})?", "", text)
    text = re.sub(r"\\end\{[^}]+\}", "", text)
    text = text.replace("&", " ")
    text = _replace_latex_command(text, "left", "")
    text = _replace_latex_command(text, "right", "")
    return text


def _split_latex_rows(text: str) -> list[str]:
    text = re.sub(r"\\begin\{[^}]+\}(?:\{[^}]*\})?", "", text)
    text = re.sub(r"\\end\{[^}]+\}", "", text)
    rows = re.split(r"\\\\", text)
    cleaned_rows = []
    for row in rows:
        row = row.replace("&", " ").strip()
        row = row.strip(",; ")
        if row:
            cleaned_rows.append(row)
    return cleaned_rows or [text]


def _parse_mathml_fragment(text: str) -> str:
    return _LatexMathMLParser(text).parse()


def _word_part(text: str) -> str:
    return _collapse_spaces(latex_to_word_linear(text))


def _replace_word_symbols(text: str) -> str:
    extra_symbols = {
        "cdots": "⋯",
        "ldots": "…",
        "to": "→",
        "rightarrow": "→",
        "leftarrow": "←",
        "Rightarrow": "⇒",
        "Leftarrow": "⇐",
        "leftrightarrow": "↔",
        "in": "∈",
        "notin": "∉",
        "subset": "⊂",
        "subseteq": "⊆",
        "supset": "⊃",
        "supseteq": "⊇",
        "cup": "∪",
        "cap": "∩",
        "forall": "∀",
        "exists": "∃",
        "angle": "∠",
        "degree": "°",
    }
    commands = {**GREEK, **SYMBOLS, **extra_symbols}
    for name, value in sorted(commands.items(), key=lambda item: -len(item[0])):
        text = _replace_latex_command(text, name, value)
    for command in (
        "displaystyle",
        "textstyle",
        "scriptstyle",
        "scriptscriptstyle",
        "limits",
        "nolimits",
    ):
        text = text.replace(f"\\{command}", "")
    return text


def _replace_latex_command(text: str, name: str, value: str) -> str:
    return re.sub(rf"\\{re.escape(name)}(?![A-Za-z])", value, text)


def _convert_one_arg_command(text: str, command: str, formatter) -> str:
    pattern = f"\\{command}"
    while True:
        index = text.find(pattern)
        if index < 0:
            return text
        group_start = _next_group_start(text, index + len(pattern))
        if group_start < 0:
            return text.replace(pattern, "", 1)
        group_end = _matching_brace(text, group_start)
        if group_end < 0:
            return text
        inner = text[group_start + 1 : group_end]
        text = text[:index] + formatter(inner) + text[group_end + 1 :]


def _convert_two_arg_command(text: str, command: str, formatter) -> str:
    pattern = f"\\{command}"
    while True:
        index = text.find(pattern)
        if index < 0:
            return text
        first_start = _next_group_start(text, index + len(pattern))
        if first_start < 0:
            return text
        first_end = _matching_brace(text, first_start)
        second_start = _next_group_start(text, first_end + 1)
        second_end = _matching_brace(text, second_start) if second_start >= 0 else -1
        if first_end < 0 or second_start < 0 or second_end < 0:
            return text
        first = text[first_start + 1 : first_end]
        second = text[second_start + 1 : second_end]
        text = text[:index] + formatter(first, second) + text[second_end + 1 :]


def _convert_scripts(text: str) -> str:
    text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_(\1)", text)
    return text


def _next_group_start(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start if start < len(text) and text[start] == "{" else -1


def _matching_brace(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _strip_remaining_latex(text: str) -> str:
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\quad", " ")
    text = text.replace("~", " ")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    return text


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraph(text: str, *, bold: bool = False) -> str:
    escaped = escape(text).replace("\n", "</w:t><w:br/><w:t>")
    bold_xml = "<w:b/>" if bold else ""
    return f"<w:p><w:r><w:rPr>{bold_xml}</w:rPr><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"


def _image_extent_emu(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    max_width_emu = int(6.0 * 914400)
    width_emu = int(width_px * 9525)
    height_emu = int(height_px * 9525)
    if width_emu > max_width_emu:
        scale = max_width_emu / width_emu
        width_emu = max_width_emu
        height_emu = int(height_emu * scale)
    return width_emu, height_emu
