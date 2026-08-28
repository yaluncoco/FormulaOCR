# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import importlib.metadata
import os
import sys

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

# Keep headroom for third-party hooks without relying on the interpreter's
# default recursion limit.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

# PyInstaller exposes SPECPATH as the directory containing this spec file,
# not the spec filename itself. Taking `.parent` here points one directory
# above the project when the spec is invoked directly.
ROOT = Path(SPECPATH).resolve()
datas = [
    (str(ROOT / 'icon.png'), '.'),
    (str(ROOT / 'icon.ico'), '.'),
]
bundled_models_root = os.environ.get('FORMULA_OCR_BUNDLED_PADDLE_MODELS', '').strip()
if bundled_models_root:
    bundled_root = Path(bundled_models_root)
    required_model_files = {'inference.json', 'inference.yml', 'inference.pdiparams'}
    if bundled_root.is_dir():
        for model_dir in bundled_root.iterdir():
            required = required_model_files
            if model_dir.name == 'LaTeX_OCR_rec':
                required = required | {'config.json'}
            if model_dir.is_dir() and required.issubset(
                {item.name for item in model_dir.iterdir() if item.is_file()}
            ):
                datas.append((str(model_dir), f'models/paddle/{model_dir.name}'))
bundled_onnx_root = os.environ.get('FORMULA_OCR_BUNDLED_ONNX_MODELS', '').strip()
onnx_model_files = {
    'RapidLaTeXOCR': {'image_resizer.onnx', 'encoder.onnx', 'decoder.onnx', 'tokenizer.json'},
    'MathCraftFormula': {
        'config.json', 'encoder_model.onnx', 'decoder_model.onnx',
        'generation_config.json', 'preprocessor_config.json',
        'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
    },
    'Pix2TextMFR15': {
        'config.json', 'encoder_model.onnx', 'decoder_model.onnx',
        'generation_config.json', 'preprocessor_config.json',
        'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
    },
    'MixTexZhEn': {
        'added_tokens.json', 'config.json', 'decoder_model_merged.onnx',
        'encoder_model.onnx', 'generation_config.json', 'merges.txt',
        'preprocessor_config.json', 'special_tokens_map.json', 'tokenizer.json',
        'tokenizer_config.json', 'vocab.json',
    },
    'UniMERNetSmallONNX': {
        'config.json', 'decoder_model_quantized.onnx',
        'decoder_with_past_model_quantized.onnx',
        'encoder_model_quantized.onnx', 'preprocessor_config.json',
        'tokenizer.json',
    },
}
if bundled_onnx_root:
    onnx_root = Path(bundled_onnx_root)
    if onnx_root.is_dir():
        for model_dir in onnx_root.iterdir():
            required = onnx_model_files.get(model_dir.name)
            if model_dir.is_dir() and required and required.issubset(
                {item.name for item in model_dir.iterdir() if item.is_file()}
            ):
                datas.append((str(model_dir), f'models/onnx/{model_dir.name}'))
binaries = []
runtime_root = Path(sys.executable).resolve().parent
runtime_bin_dirs = (
    runtime_root / 'Library' / 'bin',
    runtime_root / 'DLLs',
    runtime_root,
)
for dll_names in (
    ('tcl86t.dll',),
    ('tk86t.dll',),
    ('libexpat.dll', 'expat.dll'),
):
    dll_path = next(
        (
            directory / dll_name
            for directory in runtime_bin_dirs
            for dll_name in dll_names
            if (directory / dll_name).is_file()
        ),
        None,
    )
    if dll_path is not None:
        binaries.append((str(dll_path), '.'))

# FormulaOCR loads Paddle's native inference extension directly and does not
# execute the broad paddle package. Keep only the native CPU files proven by
# frozen PP-FormulaNet+ L inference; mklml is dynamically loaded and therefore
# must be listed even though it is not visible in the PE import table.
paddle_distribution = importlib.metadata.distribution('paddlepaddle')
paddle_root = Path(paddle_distribution.locate_file('paddle')).resolve()
libpaddle = next(
    (
        path
        for path in sorted((paddle_root / 'base').glob('libpaddle*'))
        if path.is_file() and path.suffix.lower() in {'.pyd', '.so', '.dylib'}
    ),
    None,
)
if libpaddle is None:
    raise SystemExit('paddlepaddle native inference extension was not found.')
binaries.append((str(libpaddle), 'paddle/base'))
required_paddle_libraries = {
    'common.dll',
    'libiomp5md.dll',
    'mkldnn.dll',
    'mklml.dll',
    'phi.dll',
}
for library_name in sorted(required_paddle_libraries):
    library_path = paddle_root / 'libs' / library_name
    if not library_path.is_file():
        raise SystemExit(f'Required Paddle library was not found: {library_path}')
    binaries.append((str(library_path), 'paddle/libs'))
hiddenimports = [
    'numpy',
    'tokenizers',
    'onnxruntime',
    'formula_ocr_app.paddle_formula_recognizer',
    'formula_ocr_app.model_downloader',
    'formula_ocr_app.rapid_recognizer',
    'formula_ocr_app.mathcraft_recognizer',
    'formula_ocr_app.pix2text_recognizer',
    'formula_ocr_app.mixtex_recognizer',
    'formula_ocr_app.unimernet_onnx_recognizer',
    'formula_ocr_app.rapid_model_downloader',
    'formula_ocr_app.mathcraft_model_downloader',
    'formula_ocr_app.pix2text_model_downloader',
    'formula_ocr_app.mixtex_model_downloader',
    'formula_ocr_app.unimernet_onnx_model_downloader',
    'formula_ocr_app.paddle_hf_model_downloader',
]
datas += copy_metadata('latex2mathml')
datas += collect_data_files('latex2mathml', include_py_files=False)
# Dynamic backend imports must be visible to static analysis. ONNX Runtime's
# own PyInstaller hook collects the small capi DLL/PYD set; no tools,
# quantization or transformers modules are needed by this application.
hiddenimports = list(dict.fromkeys(hiddenimports))


a = Analysis(
    [str(ROOT / 'formula_ocr_app' / 'app.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'torch',
        'torchvision',
        'torchaudio',
        'modelscope',
        'matplotlib',
        'sklearn',
        'scipy',
        'paddle',
        'paddleocr',
        'paddlex',
        'cv2',
        'pypdfium2',
        'pandas',
        'filelock',
        'sqlite3',
        '_sqlite3',
        'chardet',
        'psutil',
        'setuptools',
        'coloredlogs',
        'humanfriendly',
        'flatbuffers',
        'sympy',
        'mpmath',
        'numpy.testing',
        'numpy.distutils',
        'numpy.f2py',
        'shapely',
        'rapid_latex_ocr',
        'onnxruntime.tools',
        'onnxruntime.quantization',
        'onnxruntime.transformers',
        'PIL.AvifImagePlugin',
        'PIL.FitsImagePlugin',
        'PIL.FliImagePlugin',
        'PIL.FpxImagePlugin',
        'PIL.Hdf5StubImagePlugin',
        'PIL.MicImagePlugin',
        'PIL.MpegImagePlugin',
        'PIL.SpiderImagePlugin',
        'paddle.tensorrt',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe_options = dict(
    name='FormulaOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ROOT / 'icon.ico')],
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    contents_directory='_internal',
    **exe_options,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FormulaOCR',
)
