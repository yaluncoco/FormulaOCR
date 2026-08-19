from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree

try:
    from formula_ocr_app.formula_formats import (
        clean_recognized_latex,
        latex_to_mathml,
        mathml_to_omml,
        mathml_to_word_mathml,
    )
    from formula_ocr_app.word_clipboard import (
        FORMAT_HTML,
        FORMAT_MATHML,
        FORMAT_MATHML_PRESENTATION,
        FORMAT_OFFICE_OPEN_XML,
        build_word_clipboard_payload,
        _custom_format_bytes,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from formula_formats import (
        clean_recognized_latex,
        latex_to_mathml,
        mathml_to_omml,
        mathml_to_word_mathml,
    )
    from word_clipboard import (
        FORMAT_HTML,
        FORMAT_MATHML,
        FORMAT_MATHML_PRESENTATION,
        FORMAT_OFFICE_OPEN_XML,
        build_word_clipboard_payload,
        _custom_format_bytes,
    )


@dataclass(frozen=True)
class WordMathCase:
    name: str
    latex: str
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()


CASES = (
    WordMathCase(
        name="overline",
        latex=r"\overline{K}=C\overline{B}",
        must_contain=('<mover accent="true">', "<mi>―</mi>"),
    ),
    WordMathCase(
        name="argmax_display_limit",
        latex=r"k^{*}=\arg\max\limits_{k}J(k)",
        must_contain=("<mo>arg</mo>", "<munder><mo>max</mo>"),
        must_not_contain=(r"<mi>\arg</mi>", "<msub><mo>max</mo>"),
    ),
    WordMathCase(
        name="two_sums_keep_display_limits",
        latex=r"\sum_{i=1}^{n}\sum_{j=1}^{n} f(x_i,y_j)",
        must_contain=("<munderover>", 'largeop="true"', 'movablelimits="false"'),
        must_not_contain=("<msubsup><mo>∑</mo>",),
    ),
    WordMathCase(
        name="three_sums_keep_display_limits",
        latex=r"\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n} f(x_i,y_j,z_k)",
        must_contain=("<munderover>", 'largeop="true"', 'movablelimits="false"'),
        must_not_contain=("<msubsup><mo>∑</mo>",),
    ),
    WordMathCase(
        name="triple_sum_integral",
        latex=(
            r"\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n}"
            r" f(x_i,y_j,z_k)\Delta x\Delta y\Delta z \approx"
            r" \iiint f(x,y,z)\,dx\,dy\,dz"
        ),
        must_contain=("<munderover>", "<mo>≈</mo>"),
        must_not_contain=("<msubsup><mo>∑</mo>",),
    ),
    WordMathCase(
        name="triple_sum_in_fraction",
        latex=(
            r"\frac{\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n} a_{ijk}}"
            r"{\sum_{i=1}^{n} b_i}"
        ),
        must_contain=("<mfrac>", 'largeop="true"', 'movablelimits="false"'),
    ),
    WordMathCase(
        name="triple_product_keep_display_limits",
        latex=r"\prod_{i=1}^{n}\prod_{j=1}^{n}\prod_{k=1}^{n} p_{ijk}",
        must_contain=("<munderover>", 'largeop="true"'),
        must_not_contain=("<msubsup><mo>∏</mo>",),
    ),
    WordMathCase(
        name="covariance",
        latex=(
            r"Cov(X,Y)=\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}"
            r"(x_i-\bar{x})(y_j-\bar{y})"
        ),
        must_contain=("<mfrac>", "<munderover>", '<mover accent="true">'),
        must_not_contain=("<msubsup><mo>∑</mo>",),
    ),
    WordMathCase(
        name="mixed_overlines",
        latex=r"\bar{x}+\overline{AB}+\frac{\overline{u}}{v}",
        must_contain=('<mover accent="true">', "<mi>―</mi>", "<mfrac>"),
    ),
    WordMathCase(
        name="root_fraction_subscripts",
        latex=r"\sqrt{\frac{x_1^2+y_2^2}{z_3}}",
        must_contain=("<msqrt>", "<mfrac>", "<msubsup>"),
    ),
    WordMathCase(
        name="matrix",
        latex=r"\begin{matrix} a&b\\ c&d \end{matrix}",
        must_contain=("<mtable", "<mtr>", "<mtd>"),
    ),
    WordMathCase(name="basic_linear", latex=r"a+b=c"),
    WordMathCase(name="quadratic", latex=r"ax^2+bx+c=0"),
    WordMathCase(name="nested_fraction", latex=r"\frac{1}{1+\frac{x}{y}}"),
    WordMathCase(name="nth_root", latex=r"\sqrt[3]{x+y}"),
    WordMathCase(name="greek_symbols", latex=r"\alpha+\beta=\gamma"),
    WordMathCase(name="trig_identity", latex=r"\sin^2\theta+\cos^2\theta=1"),
    WordMathCase(name="log_exp", latex=r"\log(x)+\ln(y)+e^{i\pi}+1=0"),
    WordMathCase(name="multi_arg_function", latex=r"g\left(a,b,c\right)=h(a,b)+k(c)"),
    WordMathCase(name="interval_commas", latex=r"x\in[a,b],\ y\in(c,d)"),
    WordMathCase(name="set_builder", latex=r"A=\{x\mid x>0, x\in\mathbb{R}\}"),
    WordMathCase(name="single_sum", latex=r"\sum_{i=1}^{n} x_i"),
    WordMathCase(name="integral", latex=r"\int_{0}^{1} x^2\,dx"),
    WordMathCase(name="double_integral", latex=r"\iint_D f(x,y)\,dx\,dy"),
    WordMathCase(name="derivative", latex=r"\frac{d}{dx}f(x)=2x"),
    WordMathCase(name="partial_derivative", latex=r"\frac{\partial L}{\partial \theta_i}"),
    WordMathCase(name="hat_tilde_vec", latex=r"\hat{x}+\tilde{y}+\vec{v}"),
    WordMathCase(
        name="expectation_cov",
        latex=r"\mathrm{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]",
    ),
    WordMathCase(
        name="operatorname_softmax",
        latex=r"\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}",
    ),
    WordMathCase(name="bold_vector", latex=r"\mathbf{W}\mathbf{x}+\mathbf{b}"),
    WordMathCase(name="mathcal_loss", latex=r"\mathcal{L}=\mathcal{D}_{KL}(p\Vert q)"),
    WordMathCase(name="mathbb_space", latex=r"x\in\mathbb{R}^{n}\times\mathbb{R}^{m}"),
    WordMathCase(
        name="cases",
        latex=r"f(x)=\begin{cases}x^2,&x\ge 0\\-x,&x<0\end{cases}",
    ),
    WordMathCase(name="pmatrix", latex=r"\begin{pmatrix}a&b\\c&d\end{pmatrix}"),
    WordMathCase(name="bmatrix", latex=r"\begin{bmatrix}1&0\\0&1\end{bmatrix}"),
    WordMathCase(
        name="aligned",
        latex=r"\begin{aligned}a&=b+c\\d&=e+f\end{aligned}",
    ),
    WordMathCase(name="binomial", latex=r"\binom{n}{k}=\frac{n!}{k!(n-k)!}"),
    WordMathCase(name="relations", latex=r"a\le b\ne c\ge d\approx e"),
    WordMathCase(name="arrows", latex=r"A\to B\Rightarrow C\leftrightarrow D"),
    WordMathCase(name="logic", latex=r"\forall x\in A,\exists y\in B:x<y"),
    WordMathCase(name="norm", latex=r"\left\|x\right\|_2=\sqrt{\sum_i x_i^2}"),
    WordMathCase(name="floor_ceil", latex=r"\left\lfloor x\right\rfloor+\left\lceil y\right\rceil"),
    WordMathCase(name="angle_commas", latex=r"\left\langle x,y\right\rangle"),
    WordMathCase(name="braces_commas", latex=r"\left\{a,b,c\right\}"),
    WordMathCase(name="subscript_comma", latex=r"T_{i,j}=A_{i,j}+B_{i,j}"),
    WordMathCase(name="prime", latex=r"f'(x)+g''(x)"),
    WordMathCase(name="transpose", latex=r"\mathbf{x}^{T}\mathbf{W}\mathbf{x}"),
    WordMathCase(
        name="graph_layer",
        latex=(
            r"\mathbf{H}^{(l+1)}=\sigma\left("
            r"\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}"
            r"\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)"
        ),
    ),
    WordMathCase(
        name="attention",
        latex=r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
    ),
    WordMathCase(
        name="classification_loss",
        latex=r"\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{ic}\log\hat{y}_{ic}",
    ),
)


INLINE_DOUBLE_SUM_MATHML = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">'
    "<mrow>"
    "<msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></msubsup>"
    "<msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></msubsup>"
    "<mi>f</mi>"
    "</mrow>"
    "</math>"
)


def run_word_mathml_regression() -> list[str]:
    failures: list[str] = []
    for case in CASES:
        try:
            word_mathml = mathml_to_word_mathml(latex_to_mathml(case.latex))
            ElementTree.fromstring(word_mathml)
        except Exception as exc:
            failures.append(f"{case.name}: parse failed: {exc}")
            continue
        for expected in case.must_contain:
            if expected not in word_mathml:
                failures.append(f"{case.name}: missing {expected}")
        for forbidden in case.must_not_contain:
            if forbidden in word_mathml:
                failures.append(f"{case.name}: unexpected {forbidden}")
        for forbidden in ("fence=", "separator=", '<mspace width="0.1667em"', "<mtext>,"):
            if forbidden in word_mathml:
                failures.append(f"{case.name}: Word MathML leaked {forbidden}")
        try:
            mathml = latex_to_mathml(case.latex)
            omml = mathml_to_omml(mathml)
            ElementTree.fromstring(
                '<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
                + omml
                + "</root>"
            )
        except Exception as exc:
            failures.append(f"{case.name}: OMML parse failed: {exc}")
            continue
        if "\\sum" in case.latex and "<m:nary>" not in omml:
            failures.append(f"{case.name}: OMML missing m:nary")
        try:
            payload = build_word_clipboard_payload(mathml, plain_text=case.latex)
            ElementTree.fromstring(payload.mathml)
            fragment = _cf_html_fragment(payload.html)
            ElementTree.fromstring(fragment)
        except Exception as exc:
            failures.append(f"{case.name}: clipboard HTML parse failed: {exc}")
            continue
        formats = tuple(name for name, _data in _custom_format_bytes(payload))
        if formats != (FORMAT_HTML,):
            failures.append(f"{case.name}: unexpected clipboard formats {formats}")
        for legacy_format in (
            FORMAT_OFFICE_OPEN_XML,
            FORMAT_MATHML,
            FORMAT_MATHML_PRESENTATION,
        ):
            if legacy_format in formats:
                failures.append(f"{case.name}: legacy format still enabled")
        if payload.mathml.startswith("<?xml"):
            failures.append(f"{case.name}: HTML MathML still contains XML declaration")
        for expected in case.must_contain:
            if expected not in fragment:
                failures.append(f"{case.name}: clipboard HTML missing {expected}")
        if "<pkg:package" in fragment or "<m:oMath" in fragment:
            failures.append(f"{case.name}: clipboard HTML contains Office XML")

    try:
        word_mathml = mathml_to_word_mathml(INLINE_DOUBLE_SUM_MATHML)
        ElementTree.fromstring(word_mathml)
        payload = build_word_clipboard_payload(
            INLINE_DOUBLE_SUM_MATHML,
            plain_text=r"\sum_{i=1}^{n}\sum_{j=1}^{n} f",
        )
        fragment = _cf_html_fragment(payload.html)
    except Exception as exc:
        failures.append(f"inline_double_sum_mathml: parse failed: {exc}")
    else:
        if "<msubsup" in word_mathml or "<msubsup" in fragment:
            failures.append("inline_double_sum_mathml: still uses msubsup")
        if word_mathml.count("<munderover>") != 2:
            failures.append("inline_double_sum_mathml: missing normalized munderover")
        if 'display="block"' not in word_mathml:
            failures.append("inline_double_sum_mathml: missing block display")

    raw_quote_separator = r"F_{ctx}=Concat(F_{local}'F_{global})"
    cleaned_quote_separator = clean_recognized_latex(raw_quote_separator)
    if cleaned_quote_separator != r"F_{ctx}=Concat(F_{local},F_{global})":
        failures.append(
            "quote_separator_cleanup: failed to normalize OCR quote separator"
        )
    if clean_recognized_latex(r"f'(x)") != r"f'(x)":
        failures.append("quote_separator_cleanup: changed real prime notation")
    try:
        word_mathml = mathml_to_word_mathml(latex_to_mathml(cleaned_quote_separator))
        ElementTree.fromstring(word_mathml)
    except Exception as exc:
        failures.append(f"quote_separator_cleanup: parse failed: {exc}")
    else:
        if ">,</mo>" not in word_mathml:
            failures.append("quote_separator_cleanup: missing comma operator")
        if "′" in word_mathml or "’" in word_mathml:
            failures.append("quote_separator_cleanup: still contains quote mark")

    concat_latex = (
        r"\mathbf{F}_{\mathrm{ctx}}=\mathrm{Concat}\left("
        r"\mathbf{F}_{\mathrm{local}},\mathbf{F}_{\mathrm{global}}\right)"
    )
    try:
        word_mathml = mathml_to_word_mathml(latex_to_mathml(concat_latex))
        ElementTree.fromstring(word_mathml)
    except Exception as exc:
        failures.append(f"concat_comma_spacing: parse failed: {exc}")
    else:
        if "<mo>,</mo>" not in word_mathml:
            failures.append("concat_comma_spacing: comma is not a math operator")
        for forbidden in ("fence=", "separator=", '<mspace width="0.1667em"'):
            if forbidden in word_mathml:
                failures.append(f"concat_comma_spacing: unexpected {forbidden}")
        if "<mtext>, " in word_mathml:
            failures.append("concat_comma_spacing: still allows Word delimiter import")
    delimiter_comma_cases = {
        "paren_commas": r"\left(a,b,c\right)",
        "bracket_commas": r"\left[a,b,c\right]",
        "brace_commas": r"\left\{a,b,c\right\}",
        "angle_commas": r"\left\langle x,y\right\rangle",
    }
    for name, latex in delimiter_comma_cases.items():
        try:
            word_mathml = mathml_to_word_mathml(latex_to_mathml(latex))
            ElementTree.fromstring(word_mathml)
        except Exception as exc:
            failures.append(f"{name}: parse failed: {exc}")
            continue
        if "<mo>,</mo>" not in word_mathml:
            failures.append(f"{name}: comma is not a math operator")
        for forbidden in ("fence=", "separator=", "<mtext>,"):
            if forbidden in word_mathml:
                failures.append(f"{name}: unexpected {forbidden}")
        if "<mrow><mo>" in word_mathml and "<mo>,</mo>" in word_mathml:
            failures.append(f"{name}: still allows Word delimiter import")
    return failures


def _cf_html_fragment(payload: bytes) -> str:
    header = payload.split(b"<!doctype", 1)[0].decode("ascii", "strict")
    offsets: dict[str, int] = {}
    for line in header.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"StartFragment", "EndFragment"}:
            offsets[key] = int(value)
    start = offsets["StartFragment"]
    end = offsets["EndFragment"]
    return payload[start:end].decode("utf-8")


def main() -> None:
    failures = run_word_mathml_regression()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print(f"word-mathml-regression-ok:{len(CASES)}")


if __name__ == "__main__":
    main()
