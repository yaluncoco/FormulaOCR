# FormulaOCR

FormulaOCR 是一个本地运行的多模型公式识别工具。它通过精简的 Paddle Inference 直连后端和 ONNX Runtime 调用不同供应商的公式 OCR 模型，将图片、剪贴板图片或截图转换为可编辑 LaTeX，并提供 Office 与 Markdown 等多格式导出。运行时不依赖 PaddleOCR、PaddleX、OpenCV 或 RapidLaTeXOCR Python 包。

[![Release](https://img.shields.io/github/v/release/yaluncoco/FormulaOCR?style=flat-square)](https://github.com/yaluncoco/FormulaOCR/releases)
[![CI](https://github.com/yaluncoco/FormulaOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/yaluncoco/FormulaOCR/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D4?style=flat-square)](https://github.com/yaluncoco/FormulaOCR/releases)

## 功能

- 打开本地公式图片、粘贴剪贴板图片或框选截图。
- 粘贴新图片时会清理上一张图片的旧结果，并阻止微信等软件的临时图片路径进入 LaTeX/MathML。
- LaTeX 结果支持直接键盘编辑、撤销/重做、全选和右键菜单，修改后自动刷新 MathML 预览。
- 按钮根据屏幕缩放和文字尺寸自动适配，工具栏在窄窗口下换行；模型列表与详情支持滚动查看。
- 复制 LaTeX、选中文本或其他公式格式后，在操作附近短暂显示复制结果；Word 富格式不可用时会明确提示已复制纯文本。
- 顶部的更新图标会在启动后自动检查正式版；发现新版时显示红点，不弹窗打断操作。悬停或键盘聚焦可查看状态，点击红点查看更新说明和下载入口；无红点时点击可手动检查。
- GitHub API 限流时改用官方发布页确认最新版本，仍可显示红点，并通过发布页查看说明和下载安装包。
- 内置模型目录：12 个模型、8 个供应商，覆盖百度飞桨 `PP-FormulaNet/PP-FormulaNet+`、PaddlePaddle 官方 `LaTeX-OCR Rec`、上海 AI Lab `UniMERNet`、OpenDataLab/Cooper114 的量化 `UniMERNet Small ONNX`、RapidAI `RapidLaTeXOCR`、SakuraMathcraft `MathCraft Formula ONNX`、Breezedeus `Pix2Text MFR 1.5` 和 MixTeX `ZhEn`。
- 主界面的模型下拉框只显示已下载或随包内置的模型，用于快速切换当前识别模型，选择结果会持久保存；已删除或尚未下载的旧选择不会被隐式使用。
- “模型管理”负责浏览和下载完整模型目录，可按供应商、模型名称、使用场景和快捷标签筛选，查看状态、模型 ID、后端、用户缓存位置和下载源，并删除单个用户缓存或打开缓存目录。
- 安装包不携带模型，可在“模型管理”中按需下载并持久缓存；大模型下载可取消，已完成的断点会保留供下次继续。
- 自动清理和规范化识别出的 LaTeX；识别完成后由用户按需手动复制。
- 对 OCR 中高置信度的格式问题做确定性后处理，例如命名运算符、关系符号和 `arg max` 的下置变量；不会调用或切换到其他模型。
- 支持 LaTeX、Markdown 行内/块公式、equation 环境、MathML、OMML、HTML、AsciiMath、Typst 和 Word 线性公式。
- 为 Word 优化 MathML 粘贴格式，支持一键复制到剪贴板。
- 生成 MathML 预览图，并可导出包含公式结果的 DOCX 文件。
- 提供 Windows 桌面 GUI 和 PyInstaller 打包脚本。

## 项目结构

```text
formula_ocr_app/
  app.py                  # 应用状态、主窗口编排与程序入口
  ui_widgets.py           # 主题颜色、圆角控件、滚动条和下拉弹窗定位
  model_manager_dialog.py # 按需加载的模型浏览、筛选、下载与缓存管理窗口
  mathml_preview.py       # 按需加载的浏览器 MathML 截图与裁剪基础设施
  recognition_pipeline.py # 当前选中模型的惰性加载与单模型识别
  model_runtime.py        # 后端注册、模型缓存状态、下载和删除的统一入口
  model_api.py            # 下载进度与异常等轻量公共契约
  paddle_formula_recognizer.py # 直接调用 libpaddle 的公式推理、预处理和解码
  recognizer.py           # 旧导入路径的兼容层
  onnx_runtime.py         # ONNX provider 与 SessionOptions 的统一配置
  recognition_tests.py    # 识别后处理、下载和模型切换回归测试
  runtime_tests.py        # 并发校验、模型切换失败恢复与解码边界回归测试
  ui_tests.py             # 多档屏幕缩放、窄窗口、复制反馈与键盘交互回归测试
  model_catalog.py        # 模型供应商、场景、大小和后端目录
  app_settings.py         # 用户模型选择与条款确认持久化
  app_update.py           # GitHub 正式版查询、版本比较与可信下载地址解析
  update_tests.py         # 更新查询与版本比较回归测试
  model_downloader.py      # Paddle 官方归档下载、CRC 校验与安全解压
  download_utils.py       # 多后端共用的断点下载、SHA-256 和原子安装
  interprocess_lock.py    # 下载器共用的轻量跨进程文件锁
  paddle_hf_model_downloader.py # PaddlePaddle Hugging Face 模型清单与 SHA-256 下载
  rapid_model_downloader.py # RapidAI ONNX 多文件下载与 SHA-256 校验
  rapid_recognizer.py     # RapidLaTeXOCR ONNX Runtime 后端
  mathcraft_model_downloader.py # MathCraft ZIP 下载、清单校验与安全安装
  mathcraft_recognizer.py # MathCraft Formula 纯 ONNX Runtime 后端
  pix2text_model_downloader.py # Pix2Text MFR 1.5 多文件下载与 SHA-256 校验
  pix2text_recognizer.py # Pix2Text MFR 1.5 ONNX 后端
  mixtex_model_downloader.py # MixTeX 官方 ZIP 下载、清单校验与安全安装
  mixtex_recognizer.py     # MixTeX merged-decoder ONNX Runtime 后端
  unimernet_onnx_model_downloader.py # UniMERNet Small ONNX 固定 revision 下载与校验
  unimernet_onnx_recognizer.py # UniMERNet Small 首次/缓存 decoder 后端
  runtime_paths.py        # 安装目录内 model / app 路径
  storage_migration.py    # 旧缓存迁移、跨卷校验和失败重试
  formula_formats.py      # LaTeX/MathML/AsciiMath/Typst/Word 格式转换
  word_clipboard.py       # Word 兼容剪贴板写入
  word_clipboard_tests.py # Word MathML 回归测试
  word_paste_tests.py     # Word 粘贴相关测试
build_exe.ps1             # Windows 打包脚本
FormulaOCR.spec           # PyInstaller 唯一打包配置
run_formula_ocr.ps1       # Windows 启动脚本
requirements.txt          # Python 依赖
icon.svg / icon.png / icon.ico
```

`dist/`、`build/`、缓存、日志、临时 Word/MathML 转换验证目录以及本地第三方源码目录不纳入仓库。

## 环境

建议使用 Python 3.10。Windows 下可以使用 Conda：

```powershell
conda create -n formula_ocr python=3.10 pip -y
conda activate formula_ocr
python -m pip install -r requirements.txt
```

如果 `paddlepaddle` 默认源安装失败，可以按 Paddle 官方 CPU 轮子源安装：

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

## 运行

```powershell
python -m formula_ocr_app.app
```

或者在 Windows 上运行：

```powershell
.\run_formula_ocr.ps1
```

安装包不内置大体积模型。请先在“模型管理”中下载所需模型，再从主界面下拉框选择；下载时状态栏显示实时百分比和已下载体积。Paddle 归档使用大小、CRC32 与安全解压校验；Hugging Face、RapidAI 和 ZIP/ONNX 模型使用固定 revision 或发布文件的 SHA-256 清单校验。所有下载器共用断点续传、原子替换和跨进程锁，后续启动无需重复下载：

```text
安装目录/
  FormulaOCR.exe
  model/                    # 所有已下载模型与续传文件
    PP-FormulaNet_plus-S/
    RapidLaTeXOCR/
    ...
  app/                      # 运行库、卸载程序和辅助文件
    settings.json
    cache/
    logs/
    docs/
```

从 v1.2.0 起，模型统一保存在 EXE 同级的 `model`，依赖、设置、日志和临时文件统一保存在 `app`。安装到可写目录即可使用，不再默认向用户目录新建 FormulaOCR 缓存。升级安装会先移动旧版 `%LOCALAPPDATA%\FormulaOCR` 下的模型、设置和日志，以及旧 `_internal/models` 中的离线模型，成功后才清理旧运行库；便携包首次启动也会迁移。迁移失败会保留尚未完成的原文件供重试，不会自动重新下载；同名不同内容保存在 `app/migration-conflicts`。卸载默认保留模型与运行数据。

源码运行采用仓库根目录的同一 `model` / `app` 布局，并自动迁移 `formula_ocr_app/.cache`。如需手动迁移其他旧缓存，可运行 `FormulaOCR.exe --migrate-data --legacy-cache "旧缓存目录"`。`FORMULA_OCR_DATA_DIR` 可显式指定另一个数据根目录，其下同样使用 `model` / `app`；测试使用此覆盖时不会读取或迁移真实用户目录。该变量应指向根目录，不能指向安装目录内的 `app`、`model` 或 `_internal`。

当前目录中的模型来源为：百度飞桨模型生态中的 PP-FormulaNet 系列与 UniMERNet、PaddlePaddle 官方 Hugging Face 的 `LaTeX_OCR_rec`、OpenDataLab/Cooper114 的量化 `UniMERNet Small ONNX`、RapidAI 社区的 RapidLaTeXOCR、SakuraMathcraft 的 MathCraft Formula ONNX，以及 Breezedeus 的 Pix2Text MFR 1.5 ONNX。PP-FormulaNet/UniMERNet 的旧 `paddlex/official_models` 缓存会迁移到 `model`；推理直接加载 `libpaddle`，不会导入 PaddleOCR/PaddleX。`LaTeX_OCR_rec` 固定到 Hugging Face revision `563fb029dfdf5fc847d0677f3870039960e3a801`，四个推理文件逐文件验证 SHA-256；RapidLaTeXOCR 的四个官方 release 文件、MathCraft ZIP 内文件、Pix2Text 固定 revision 的八个文件和 UniMERNet Small ONNX 固定 revision 的六个文件也均记录并验证 SHA-256；Paddle 归档则按官方归档大小和 CRC32 校验。识别时只调用主界面明确选择的模型，不会跨供应商自动切换。UniMERNet Small ONNX 的原始模型来源为 `wanderkid/unimernet_small`，转换仓库为 `Cooper114/unimernet-onnx`，模型卡标注 Apache-2.0。
当前目录还提供 MixTeX 官方 `MixTeX-v3.2.4` 可选模型。它使用官方 `MixTeX.zip` 中的 `encoder_model.onnx`、`decoder_model_merged.onnx` 和 tokenizer 文件，ZIP 大小为 294,025,378 字节，按 SHA-256 `734088e8c3ac6d0ebf02b3054ed0cdde7d8be2eb57c33b8f049a66d05e026750` 校验。MixTeX 上游条款标注 AGPL-3.0，并声明基于该模型的衍生品不得用于商业用途；FormulaOCR 首次下载/使用前会在界面提示并记录条款确认。若你的软件或离线包有商业分发计划，请不要分发 MixTeX 权重，或先取得上游书面授权。

每次识别只调用主界面当前选择的一个模型；程序不会因识别结果自动切换模型，也不会隐式下载 M、L 或其他候选模型。源码运行和安装版使用相同的目录规则。

## 下载与安装

Windows 用户可以直接从 [GitHub Releases](https://github.com/yaluncoco/FormulaOCR/releases) 下载：

```text
FormulaOCRSetup-1.2.0.exe
```

安装程序是 x64 Windows 的 Inno Setup 安装包，默认安装到当前用户目录，不需要管理员权限。安装后可以从开始菜单启动 FormulaOCR；桌面快捷方式在安装时可选。卸载时默认保留模型、设置和日志，避免重新安装后重复下载；如果确认不再需要，也可以选择同时清理安装目录中的 `model` 与 `app` 运行数据。

安装包不包含大体积模型权重。第一次使用某个模型时，程序会从模型管理中的官方地址下载并校验模型，模型与续传文件保存在安装目录的 `model`。这样更新程序时无需重新打包或覆盖本地模型。

每个 Release 同时提供 `.sha256` 校验文件。下载后可在 PowerShell 中验证：

```powershell
Get-FileHash .\FormulaOCRSetup-1.2.0.exe -Algorithm SHA256
Get-Content .\FormulaOCRSetup-1.2.0.exe.sha256
```

当前公开 Release 安装程序未进行商业代码签名，Windows SmartScreen 在下载量较少时可能显示“未知发布者”。请仅从本项目 Releases 下载，并用 SHA-256 文件核对完整性。

## 使用

1. 打开程序。
2. 点击打开图片、粘贴图片或截图。
3. 先在“模型管理”中下载需要的模型，再从主界面下拉框选择已下载模型。
4. 点击识别。
5. 在右侧查看 LaTeX 和格式转换结果。
6. 按需复制 LaTeX、复制 Word MathML、复制其他格式或导出 DOCX。

## 打包

Windows 下可以先生成 PyInstaller 目录版：

```powershell
.\build_exe.ps1
```

脚本固定生成启动较快的目录版；发布时必须复制整个 `dist\FormulaOCR` 文件夹，不能只复制其中的 EXE。

打包脚本统一读取 `FormulaOCR.spec`，完成文件复制后会自动运行打包版缓存边界、UI 和 Word/MathML 回归自检；如果 EXE、界面或关键运行库无法启动，脚本会直接失败。

脚本默认查找 `C:\D\anaconda3\envs\formula_ocr` 环境。如果 Conda 安装路径不同，可先设置 `FORMULA_OCR_CONDA_ENV`：

```powershell
$env:FORMULA_OCR_CONDA_ENV = "D:\anaconda3\envs\formula_ocr"
.\build_exe.ps1
```

要生成与 Releases 相同的单文件安装程序，请安装 Inno Setup 6 后运行：

```powershell
.\build_installer.ps1
```

`build_installer.ps1` 会先构建并自检 `dist\FormulaOCR`，再生成 `dist\installer\FormulaOCRSetup-1.2.0.exe` 和对应 SHA-256 文件。公开仓库的 `.github/workflows/release.yml` 会在推送 `v*` 标签时于干净的 Windows runner 上先运行回归测试，再根据 `requirements.txt` 重复构建流程并上传 Release 草稿资产，验证后发布；不需要把第三方 OCR 源码或模型权重提交到仓库。

打包产物会输出到：

```text
dist/FormulaOCR/
```

上传到 GitHub 时不要提交 `dist/` 和 `build/`，它们体积很大且可以重新生成。

打包脚本不会再把本机模型缓存复制进 `dist/`，因此发布包体积更小，但用户首次识别时需要联网下载所选模型。

如需制作离线包，可将包含各模型完整推理文件的父目录设置到 `FORMULA_OCR_BUNDLED_PADDLE_MODELS` 后再打包；脚本将模型装入 `app\models\paddle\<模型名>`，安装迁移后统一存入 `model\<模型名>`。由于全部 Paddle 模型约 4GB 以上，这个选项默认关闭。

ONNX 模型也支持可选的只读随包方式：将 `RapidLaTeXOCR`、`MathCraftFormula`、`Pix2TextMFR15`、`UniMERNetSmallONNX` 或已获授权的 `MixTexZhEn` 子目录放在 `FORMULA_OCR_BUNDLED_ONNX_MODELS` 指定的父目录中，再运行打包脚本；完整文件先放到 `app\models\onnx\<模型名>`，安装迁移后统一存入 `model\<模型名>`；程序继续校验模型完整性，复用已有文件。MixTeX 仍受其上游非商业条款约束，默认不应随商业安装包分发。

## 测试

```powershell
python -m formula_ocr_app.app --word-mathml-self-test
python -m formula_ocr_app.app --clipboard-self-test
python -m formula_ocr_app.app --runtime-self-test
python -m formula_ocr_app.app --preview-self-test
python -m formula_ocr_app.app --ui-self-test
# 已下载并确认 MixTeX 条款后，可做真实 ONNX smoke test
python -m formula_ocr_app.app --self-test --self-test-model MixTexZhEn
python -m formula_ocr_app.recognition_tests
python -m formula_ocr_app.update_tests
python -m formula_ocr_app.runtime_tests
python -m formula_ocr_app.ui_tests
```

部分测试依赖 Windows 剪贴板、Word 兼容格式或本地浏览器。

CI 会合并运行四个 unittest 模块，并检查每项桌面自检的退出码，任一步失败都会终止该步骤。运行环境自检需要单独启动 Python 进程，以检查启动阶段的惰性加载。

性能改动、测量范围与后续改进建议见 [2026-09 优化审查记录](docs/optimization-2026-09.md)。

v1.2.0 新增统一存储和旧模型自动迁移、ONNX 线程控制、后台预览转换及图片重绘优化，详见 [本次验证记录](docs/storage-performance-2026-10.md)。此前版本包含紧凑的更新图标、新版红点提醒及 GitHub API 限流时的发布页备用检查，保留缩放适配、复制提示、百分比区间识别和小屏幕布局修复，详见 [界面与交互检查记录](docs/ui-review-2026-09.md) 与 [数字区间识别检查记录](docs/recognition-review-2026-09.md)。

## 开源致谢

本项目的公式识别能力基于 PaddlePaddle/Paddle 模型生态、PaddlePaddle 官方的 [`LaTeX_OCR_rec`](https://huggingface.co/PaddlePaddle/LaTeX_OCR_rec)、[UniMERNet Small ONNX](https://huggingface.co/Cooper114/unimernet-onnx)、[RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR)、[MathCraft-Models](https://github.com/SakuraMathcraft/MathCraft-Models)、[Pix2Text MFR](https://huggingface.co/breezedeus/pix2text-mfr-1.5) 和 [MixTeX](https://github.com/RQLuo/MixTeX-Latex-OCR)。Paddle 预处理/解码行为参考了 Apache-2.0 的 PaddleX 公式识别处理器，但发布程序不携带 PaddleOCR/PaddleX Python 包；Rapid 后端也只使用其官方模型资产，不依赖 RapidLaTeXOCR Python 包。MathCraft Formula 的 ONNX 推理实现参考了 [LaTeXSnipper](https://github.com/SakuraMathcraft/LaTeXSnipper) 的模型适配思路。

本仓库不包含第三方源码、模型权重或打包后的运行时文件。若你在二进制包或离线包中一并分发 PaddlePaddle 运行库或 PP-FormulaNet、UniMERNet、RapidLaTeXOCR、MathCraft、Pix2Text、MixTeX 等模型文件，请同时保留对应项目的许可证、版权声明和模型使用说明；MixTeX 还必须遵守其非商业限制。更完整的第三方声明见 [NOTICE.md](NOTICE.md)。

## 说明

本仓库只保存应用逻辑、界面代码、格式转换代码和必要资源。模型文件、第三方源码、打包后的可执行文件、日志和临时转换验证文件应通过 `.gitignore` 排除。
