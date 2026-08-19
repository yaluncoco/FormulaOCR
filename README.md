# FormulaOCR

FormulaOCR 是一个本地运行的多模型公式识别工具。它通过 Paddle/PaddleX 和 ONNX Runtime 调用不同供应商的公式 OCR 模型，将图片、剪贴板图片或截图转换为可编辑 LaTeX，并提供 Office 与 Markdown 等多格式导出。

[![Release](https://img.shields.io/github/v/release/yaluncoco/FormulaOCR?style=flat-square)](https://github.com/yaluncoco/FormulaOCR/releases)
[![CI](https://github.com/yaluncoco/FormulaOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/yaluncoco/FormulaOCR/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D4?style=flat-square)](https://github.com/yaluncoco/FormulaOCR/releases)

## 功能

- 打开本地公式图片、粘贴剪贴板图片或框选截图。
- 内置模型目录：12 个模型、8 个供应商，覆盖百度飞桨 `PP-FormulaNet/PP-FormulaNet+`、PaddlePaddle 官方 `LaTeX-OCR Rec`、上海 AI Lab `UniMERNet`、OpenDataLab/Cooper114 的量化 `UniMERNet Small ONNX`、RapidAI `RapidLaTeXOCR`、SakuraMathcraft `MathCraft Formula ONNX`、Breezedeus `Pix2Text MFR 1.5` 和 MixTeX `ZhEn`。
- 主界面的模型下拉框只显示已下载或随包内置的模型，用于快速切换当前识别模型，选择结果会持久保存。
- “模型管理”负责浏览和下载完整模型目录，可按供应商、模型名称、使用场景和快捷标签筛选，查看状态、模型 ID、后端、用户缓存位置和下载源，并删除单个用户缓存或打开缓存目录。
- 安装包不携带模型，首次使用时在软件内按需下载并持久缓存；大模型下载可取消，已完成的断点会保留供下次继续。
- 自动清理和规范化识别出的 LaTeX；识别完成后由用户按需手动复制。
- 对 OCR 中高置信度的格式问题做确定性后处理，例如命名运算符、关系符号和 `arg max` 的下置变量；不会调用或切换到其他模型。
- 支持 LaTeX、Markdown 行内/块公式、equation 环境、MathML、OMML、HTML、AsciiMath、Typst 和 Word 线性公式。
- 为 Word 优化 MathML 粘贴格式，支持一键复制到剪贴板。
- 生成 MathML 预览图，并可导出包含公式结果的 DOCX 文件。
- 提供 Windows 桌面 GUI 和 PyInstaller 打包脚本。

## 项目结构

```text
formula_ocr_app/
  app.py                  # Tkinter 桌面界面和应用入口
  recognizer.py           # PaddleOCR 公式识别封装
  recognition_pipeline.py # 当前选中模型的惰性加载与单模型识别
  recognition_tests.py    # 识别后处理、下载和模型切换回归测试
  model_catalog.py        # 模型供应商、场景、大小和后端目录
  app_settings.py         # 用户模型选择与条款确认持久化
  model_downloader.py      # 官方模型断点下载、CRC 校验与安全解压
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
  runtime_paths.py        # 用户缓存、日志和按需下载模型路径
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

`dist/`、`build/`、缓存、日志、临时 Word/MathML 转换验证目录以及第三方 PaddleOCR 源码目录不纳入仓库。

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
python -m pip install "paddlex[ocr-core]>=3.6.0,<3.7.0" PyYAML requests aiohttp tokenizers ftfy latex2mathml pillow
```

## 运行

```powershell
python -m formula_ocr_app.app
```

或者在 Windows 上运行：

```powershell
.\run_formula_ocr.ps1
```

安装包不内置大体积模型。首次使用某个模型时，程序会从该模型的官方发布源后台下载，状态栏显示实时百分比和已下载体积。Paddle 模型使用 CRC32 校验与安全解压，RapidAI ONNX 模型逐文件使用 SHA-256 校验，二者均支持断点续传。下载完成后自动继续识别，后续启动无需重复下载：

```text
%LOCALAPPDATA%\FormulaOCR\cache\runtime\paddlex\official_models
%LOCALAPPDATA%\FormulaOCR\cache\runtime\paddlex\official_models\LaTeX_OCR_rec
%LOCALAPPDATA%\FormulaOCR\cache\models\RapidLaTeXOCR
%LOCALAPPDATA%\FormulaOCR\cache\models\MathCraftFormula
%LOCALAPPDATA%\FormulaOCR\cache\models\Pix2TextMFR15
%LOCALAPPDATA%\FormulaOCR\cache\models\MixTexZhEn
%LOCALAPPDATA%\FormulaOCR\cache\models\UniMERNetSmallONNX
```

这里的缓存策略与 LaTeXSnipper 的思路一致，但要区分两类目录：`dist\FormulaOCR\_internal` 是 PyInstaller 目录版的随包运行库目录，适合放程序代码、DLL 或明确随安装包分发的只读模型；它不适合运行时下载，因为升级覆盖和安装目录权限限制都可能让下载失败。LaTeXSnipper 的 Windows 包本身也是 `onedir`，安装时会带上基础 `_internal`；它另外下载的是独立 OCR 依赖层和用户模型，并不是在缺少 `_internal` 时补齐主程序。FormulaOCR 将运行时模型放在上面的用户数据目录，下载临时文件、校验和安装也都在那里完成。这样既可以发布一个不带模型的安装包，也可以在打包时把经过授权的离线模型放进随包资源而不污染用户缓存。

冻结版即使设置了 `FORMULA_OCR_DATA_DIR`，只要该路径落在 EXE 或 `_internal` 内，程序也会拒绝把它作为可写缓存并回退到 `%LOCALAPPDATA%\FormulaOCR\cache`；这避免了误把 PyInstaller 运行库目录当成模型下载目录。

当前目录中的模型来源为：百度飞桨/PaddleX 的 PP-FormulaNet 系列与 UniMERNet、PaddlePaddle 官方 Hugging Face 的 `LaTeX_OCR_rec`、OpenDataLab/Cooper114 的量化 `UniMERNet Small ONNX`、RapidAI 社区的 RapidLaTeXOCR、SakuraMathcraft 的 MathCraft Formula ONNX，以及 Breezedeus 的 Pix2Text MFR 1.5 ONNX。`LaTeX_OCR_rec` 固定到 Hugging Face revision `563fb029dfdf5fc847d0677f3870039960e3a801`，四个推理文件逐文件验证 SHA-256；RapidLaTeXOCR 的四个官方 release 文件、MathCraft ZIP 内文件、Pix2Text 固定 revision 的八个文件和 UniMERNet Small ONNX 固定 revision 的六个文件也均记录并验证 SHA-256；Paddle 归档则按官方归档大小和 CRC32 校验。MathCraft 是与 LaTeXSnipper 同源的 ONNX 模型资产，Pix2Text MFR 1.5 和 UniMERNet Small ONNX 来自官方 Hugging Face 模型卡；它们都使用独立后端。识别时只调用主界面明确选择的模型，不会跨供应商自动切换。UniMERNet Small ONNX 的原始模型来源为 `wanderkid/unimernet_small`，转换仓库为 `Cooper114/unimernet-onnx`，模型卡标注 Apache-2.0。后续增加供应商时，只需补充模型目录、下载清单和对应推理后端，不应直接把未经校验的权重复制进 `_internal`。
当前目录还提供 MixTeX 官方 `MixTeX-v3.2.4` 可选模型。它使用官方 `MixTeX.zip` 中的 `encoder_model.onnx`、`decoder_model_merged.onnx` 和 tokenizer 文件，ZIP 大小为 294,025,378 字节，按 SHA-256 `734088e8c3ac6d0ebf02b3054ed0cdde7d8be2eb57c33b8f049a66d05e026750` 校验。MixTeX 上游条款标注 AGPL-3.0，并声明基于该模型的衍生品不得用于商业用途；FormulaOCR 首次下载/使用前会在界面提示并记录条款确认。若你的软件或离线包有商业分发计划，请不要分发 MixTeX 权重，或先取得上游书面授权。

每次识别只调用主界面当前选择的一个模型；程序不会因识别结果自动切换模型，也不会隐式下载 M、L 或其他候选模型。源码运行时仍使用 `formula_ocr_app/.cache`。可以通过 `FORMULA_OCR_DATA_DIR` 环境变量指定其他数据目录。

## 下载与安装

Windows 用户可以直接从 [GitHub Releases](https://github.com/yaluncoco/FormulaOCR/releases) 下载：

```text
FormulaOCRSetup-1.0.0.exe
```

安装程序是 x64 Windows 的 Inno Setup 安装包，默认安装到当前用户目录，不需要管理员权限。安装后可以从开始菜单启动 FormulaOCR；桌面快捷方式在安装时可选。卸载时默认保留模型、设置和日志，避免重新安装后重复下载；如果确认不再需要，也可以选择同时清理 `%LOCALAPPDATA%\FormulaOCR`。

安装包不包含大体积模型权重。第一次使用某个模型时，程序会从模型管理中的官方地址下载并校验模型，下载进度和断点保存在 `%LOCALAPPDATA%\FormulaOCR\cache`。这样更新程序时无需重新打包或覆盖本地模型。

每个 Release 同时提供 `.sha256` 校验文件。下载后可在 PowerShell 中验证：

```powershell
Get-FileHash .\FormulaOCRSetup-1.0.0.exe -Algorithm SHA256
Get-Content .\FormulaOCRSetup-1.0.0.exe.sha256
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

`build_installer.ps1` 会先构建并自检 `dist\FormulaOCR`，再生成 `dist\installer\FormulaOCRSetup-1.0.0.exe` 和对应 SHA-256 文件。公开仓库的 `.github/workflows/release.yml` 会在推送 `v*` 标签时于干净的 Windows runner 上重复这个流程并自动上传 Release 资产；构建环境固定安装官方 `paddleocr==3.6.0` 包，不需要把第三方源码提交到仓库。

打包产物会输出到：

```text
dist/FormulaOCR/
```

上传到 GitHub 时不要提交 `dist/` 和 `build/`，它们体积很大且可以重新生成。

打包脚本不会再把本机模型缓存复制进 `dist/`，因此发布包体积更小，但用户首次识别时需要联网下载所选模型。

如需制作离线包，可将包含各模型完整推理文件的父目录设置到 `FORMULA_OCR_BUNDLED_PADDLE_MODELS` 后再打包；脚本会把其中的模型放到 `_internal\models\paddle\<模型名>`，程序优先读取随包模型，并仍把新下载内容写入用户数据目录。由于全部 Paddle 模型约 4GB 以上，这个选项默认关闭。

ONNX 模型也支持可选的只读随包方式：将 `RapidLaTeXOCR`、`MathCraftFormula`、`Pix2TextMFR15`、`UniMERNetSmallONNX` 或已获授权的 `MixTexZhEn` 子目录放在 `FORMULA_OCR_BUNDLED_ONNX_MODELS` 指定的父目录中，再运行打包脚本；完整文件会放到 `_internal\models\onnx\<模型名>`。程序会先使用已校验的用户缓存，再使用已校验的随包模型，最后才下载。运行时下载不会改写 `_internal`。MixTeX 仍受其上游非商业条款约束，默认不应随商业安装包分发。

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
```

部分测试依赖 Windows 剪贴板、Word 兼容格式或本地浏览器。

## 开源致谢

本项目的公式识别能力基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / PaddleX 公式识别模型、PaddlePaddle 官方的 [`LaTeX_OCR_rec`](https://huggingface.co/PaddlePaddle/LaTeX_OCR_rec)、[UniMERNet Small ONNX](https://huggingface.co/Cooper114/unimernet-onnx)、[RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR)、[MathCraft-Models](https://github.com/SakuraMathcraft/MathCraft-Models)、[Pix2Text MFR](https://huggingface.co/breezedeus/pix2text-mfr-1.5) 和 [MixTeX](https://github.com/RQLuo/MixTeX-Latex-OCR)。MathCraft Formula 的 ONNX 推理实现参考了 [LaTeXSnipper](https://github.com/SakuraMathcraft/LaTeXSnipper) 的模型适配思路；模型权重仍分别从各自官方 release 或固定的 Hugging Face revision 下载并校验。

本仓库不包含第三方源码、模型权重或打包后的运行时文件。若你在二进制包或离线包中一并分发 PaddleOCR、PaddlePaddle、PaddleX、RapidLaTeXOCR、UniMERNet、MathCraft、Pix2Text 或 MixTeX 模型文件，请同时保留对应项目的许可证、版权声明和模型使用说明；MixTeX 还必须遵守其非商业限制。更完整的第三方声明见 [NOTICE.md](NOTICE.md)。

## 说明

本仓库只保存应用逻辑、界面代码、格式转换代码和必要资源。模型文件、PaddleOCR 第三方源码、打包后的可执行文件、日志和临时转换验证文件应通过 `.gitignore` 排除。
