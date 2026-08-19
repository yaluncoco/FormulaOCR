from __future__ import annotations

from dataclasses import dataclass


MODEL_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/"
    "paddlex/official_inference_model/paddle3.0.0"
)
RAPID_RELEASE_URL = (
    "https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0"
)
RAPID_RELEASE_PAGE_URL = "https://github.com/RapidAI/RapidLaTeXOCR/releases/tag/v0.0.0"
MATHCRAFT_RELEASE_URL = (
    "https://github.com/SakuraMathcraft/MathCraft-Models/releases/download/v1.0.0"
)
MATHCRAFT_LICENSE_URL = (
    "https://github.com/SakuraMathcraft/MathCraft-Models/blob/main/LICENSE"
)
MIXTEX_RELEASE_PAGE_URL = (
    "https://github.com/RQLuo/MixTeX-Latex-OCR/releases/tag/MixTeX-v3.2.4"
)
MIXTEX_TERMS_URL = (
    "https://github.com/RQLuo/MixTeX-Latex-OCR/blob/main/"
    "User%20Manual%26Terms%20of%20Service.md"
)
PIX2TEXT_MODEL_URL = "https://huggingface.co/breezedeus/pix2text-mfr-1.5"
PADDLE_HF_MODEL_URL = "https://huggingface.co/PaddlePaddle/LaTeX_OCR_rec"
UNIMERNET_ONNX_MODEL_PAGE_URL = (
    "https://huggingface.co/Cooper114/unimernet-onnx/tree/"
    "411ee76221baaad144ffbf996d4deef8df013b54/small"
)

LIGHTWEIGHT_MODEL_MAX_BYTES = 400 * 1024 * 1024
MODEL_QUICK_FILTERS = (
    ("all", "全部"),
    ("recommended", "推荐"),
    ("downloaded", "已下载"),
    ("lightweight", "轻量"),
    ("chinese", "中文"),
    ("onnx", "ONNX"),
)
MODEL_BACKEND_LABELS = {
    "paddle": "Paddle / PaddleX",
    "paddle_hf": "Paddle + Hugging Face",
    "rapid_onnx": "ONNX Runtime / RapidLaTeXOCR",
    "mathcraft_onnx": "ONNX Runtime / MathCraft",
    "pix2text_onnx": "ONNX Runtime / Pix2Text",
    "mixtex_onnx": "ONNX Runtime / MixTeX",
    "unimernet_onnx": "ONNX Runtime / UniMERNet Small",
}


@dataclass(frozen=True)
class FormulaModelSpec:
    model_id: str
    display_name: str
    provider: str
    family: str
    archive_size: int
    archive_crc32: int | None
    description: str
    best_for: str
    languages: str
    backend: str = "paddle"
    license_label: str = ""
    terms_url: str = ""
    usage_restriction: str = ""
    terms_revision: str = ""
    recommended: bool = False

    @property
    def archive_name(self) -> str:
        if self.backend == "mathcraft_onnx":
            return "mathcraft-formula-rec.zip"
        if self.backend == "pix2text_onnx":
            return "pix2text-mfr-1.5"
        if self.backend == "mixtex_onnx":
            return "MixTeX.zip"
        if self.backend == "unimernet_onnx":
            return "unimernet-small-onnx"
        if self.backend == "paddle_hf":
            return "LaTeX_OCR_rec"
        return f"{self.model_id}_infer.tar"

    @property
    def archive_root(self) -> str:
        return f"{self.model_id}_infer"

    @property
    def download_url(self) -> str:
        if self.backend == "rapid_onnx":
            # RapidLaTeXOCR is a four-file release rather than one archive;
            # expose the release page as the catalog source while the
            # dedicated downloader resolves and verifies each file.
            return RAPID_RELEASE_PAGE_URL
        if self.backend == "mixtex_onnx":
            return MIXTEX_RELEASE_PAGE_URL
        if self.backend == "mathcraft_onnx":
            return f"{MATHCRAFT_RELEASE_URL}/{self.archive_name}"
        if self.backend == "pix2text_onnx":
            return PIX2TEXT_MODEL_URL
        if self.backend == "unimernet_onnx":
            return UNIMERNET_ONNX_MODEL_PAGE_URL
        if self.backend == "paddle_hf":
            return PADDLE_HF_MODEL_URL
        return f"{MODEL_BASE_URL}/{self.archive_name}"

    @property
    def uses_paddle_runtime(self) -> bool:
        return self.backend in {"paddle", "paddle_hf"}

    @property
    def is_onnx(self) -> bool:
        return self.backend.endswith("_onnx")

    @property
    def is_lightweight(self) -> bool:
        return self.archive_size <= LIGHTWEIGHT_MODEL_MAX_BYTES

    @property
    def backend_label(self) -> str:
        return MODEL_BACKEND_LABELS.get(self.backend, self.backend)

    @property
    def searchable_text(self) -> str:
        return " ".join(
            (
                self.model_id,
                self.display_name,
                self.provider,
                self.family,
                self.best_for,
                self.languages,
                self.description,
                self.backend,
                self.backend_label,
                self.license_label,
                self.usage_restriction,
            )
        ).casefold()

    @property
    def requires_terms_ack(self) -> bool:
        return bool(self.terms_url and self.usage_restriction)

    @property
    def download_mebibytes(self) -> float:
        return self.archive_size / (1024 * 1024)

    @property
    def size_label(self) -> str:
        return f"约 {self.download_mebibytes:.0f} MB"

    @property
    def compact_name(self) -> str:
        """Short label used by the compact model picker."""

        return self.display_name.split("（", 1)[0].strip()

    @property
    def selector_label(self) -> str:
        return f"{self.provider} · {self.display_name} · {self.size_label}"


MODEL_SPECS = (
    FormulaModelSpec(
        model_id="PP-FormulaNet_plus-S",
        display_name="PP-FormulaNet+ S（推荐）",
        provider="百度飞桨",
        family="PP-FormulaNet+",
        archive_size=259_604_480,
        archive_crc32=499_135_057,
        description="速度快、中文公式能力较好，适合日常识别。",
        best_for="日常截图、中文/英文混合公式",
        languages="中文、英文",
        recommended=True,
    ),
    FormulaModelSpec(
        model_id="PP-FormulaNet_plus-M",
        display_name="PP-FormulaNet+ M",
        provider="百度飞桨",
        family="PP-FormulaNet+",
        archive_size=620_451_840,
        archive_crc32=1_382_885_086,
        description="准确率与速度平衡，适合作为默认高精度模型。",
        best_for="复杂印刷公式、中文公式",
        languages="中文、英文",
    ),
    FormulaModelSpec(
        model_id="PP-FormulaNet_plus-L",
        display_name="PP-FormulaNet+ L",
        provider="百度飞桨",
        family="PP-FormulaNet+",
        archive_size=731_504_640,
        archive_crc32=3_394_071_220,
        description="PP-FormulaNet+ 最高精度档，CPU 初始化和推理较慢。",
        best_for="高难度及长公式、中文公式",
        languages="中文、英文",
    ),
    FormulaModelSpec(
        model_id="PP-FormulaNet-S",
        display_name="PP-FormulaNet S",
        provider="百度飞桨",
        family="PP-FormulaNet",
        archive_size=234_434_560,
        archive_crc32=2_483_476_691,
        description="原始轻量系列，英文公式速度优先。",
        best_for="英文印刷公式、低配置电脑",
        languages="英文为主",
    ),
    FormulaModelSpec(
        model_id="PP-FormulaNet-L",
        display_name="PP-FormulaNet L",
        provider="百度飞桨",
        family="PP-FormulaNet",
        archive_size=728_360_960,
        archive_crc32=978_989_952,
        description="原始高精度系列，英文复杂公式表现稳定。",
        best_for="复杂英文公式、手写公式",
        languages="英文为主",
    ),
    FormulaModelSpec(
        model_id="UniMERNet",
        display_name="UniMERNet",
        provider="上海 AI Lab",
        family="UniMERNet",
        archive_size=1_639_956_480,
        archive_crc32=2_753_224_158,
        description="真实场景通用模型，体积大，适合交叉验证及手写公式。",
        best_for="真实拍照、复杂排版、手写公式",
        languages="英文、部分中文",
    ),
    FormulaModelSpec(
        model_id="UniMERNetSmallONNX",
        display_name="UniMERNet Small ONNX（量化）",
        provider="OpenDataLab / Cooper114",
        family="UniMERNet",
        archive_size=349_928_844,
        archive_crc32=None,
        description=(
            "UniMERNet Small 的量化 ONNX 导出，独立于 Paddle 运行时；"
            "体积和启动成本低于完整 UniMERNet，适合本地 CPU 推理。"
        ),
        best_for="真实场景、复杂排版、手写公式的轻量识别",
        languages="英文、部分中文",
        backend="unimernet_onnx",
        recommended=True,
    ),
    FormulaModelSpec(
        model_id="LaTeX_OCR_rec",
        display_name="LaTeX-OCR Rec（轻量）",
        provider="PaddlePaddle / Hugging Face",
        family="LaTeX-OCR",
        archive_size=103_735_029,
        archive_crc32=None,
        description=(
            "PaddlePaddle 官方 LaTeX-OCR 自回归模型，体积较小；"
            "适合英文/中文基础公式和轻量部署。"
        ),
        best_for="轻量部署、基础印刷公式、中文/英文公式",
        languages="中文、英文",
        backend="paddle_hf",
    ),
    FormulaModelSpec(
        model_id="RapidLaTeXOCR",
        display_name="RapidLaTeXOCR / pix2tex",
        provider="RapidAI 社区",
        family="LaTeX-OCR",
        archive_size=178_952_787,
        archive_crc32=None,
        description="社区经典 pix2tex 路线的 ONNX Runtime 版本，适合 Windows CPU。",
        best_for="英文印刷公式、轻量备选",
        languages="英文为主",
        backend="rapid_onnx",
    ),
    FormulaModelSpec(
        model_id="MathCraftFormula",
        display_name="MathCraft Formula ONNX",
        provider="SakuraMathcraft",
        family="MathCraft OCR",
        archive_size=108_795_631,
        archive_crc32=None,
        description="LaTeXSnipper 同源的纯 ONNX 公式识别模型，独立于 Paddle 运行时。",
        best_for="英文印刷公式、论文截图、CPU/GPU ONNX 推理",
        languages="英文为主",
        backend="mathcraft_onnx",
        license_label="MIT",
        terms_url=MATHCRAFT_LICENSE_URL,
    ),
    FormulaModelSpec(
        model_id="Pix2TextMFR15",
        display_name="Pix2Text MFR 1.5",
        provider="Breezedeus / Pix2Text",
        family="Pix2Text MFR",
        archive_size=119_654_633,
        archive_crc32=None,
        description="Pix2Text 1.5 公开 MFR ONNX 模型，适合印刷体和复杂公式；独立于 Paddle。",
        best_for="英文印刷公式、矩阵、长公式",
        languages="英文为主",
        backend="pix2text_onnx",
        recommended=True,
    ),
    FormulaModelSpec(
        model_id="MixTexZhEn",
        display_name="MixTeX ZhEn（中英混合）",
        provider="MixTeX / RQLuo",
        family="MixTeX LaTeX OCR",
        archive_size=294_025_378,
        archive_crc32=None,
        description=(
            "MixTeX 官方 CPU ONNX release，支持中英文混合公式、部分手写内容和简单表格；"
            "使用 merged decoder 进行低依赖推理。"
        ),
        best_for="中英文混合公式、手写公式、简单表格",
        languages="中文、英文",
        backend="mixtex_onnx",
        license_label="AGPL-3.0（含上游非商业限制）",
        terms_url=MIXTEX_TERMS_URL,
        usage_restriction="上游条款声明：基于该模型的衍生品不得用于商业用途。",
        terms_revision="MixTeX-v3.2.4",
    ),
)

MODEL_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}
MODEL_BY_LABEL = {spec.selector_label: spec for spec in MODEL_SPECS}
DEFAULT_MODEL_ID = "PP-FormulaNet_plus-S"


def get_model_spec(model_id: str) -> FormulaModelSpec:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"未知公式识别模型：{model_id}") from exc


def model_matches_query(spec: FormulaModelSpec, query: str) -> bool:
    normalized = query.strip().casefold()
    return not normalized or normalized in spec.searchable_text


def model_matches_quick_filter(
    spec: FormulaModelSpec,
    filter_key: str,
    *,
    cached: bool = False,
) -> bool:
    if filter_key == "all":
        return True
    if filter_key == "recommended":
        return spec.recommended
    if filter_key == "downloaded":
        return cached
    if filter_key == "lightweight":
        return spec.is_lightweight
    if filter_key == "chinese":
        return "中文" in spec.languages
    if filter_key == "onnx":
        return spec.is_onnx
    raise ValueError(f"未知模型快捷筛选：{filter_key}")
