# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

# PyInstaller exposes SPECPATH as the directory containing this spec file,
# not the spec filename itself. Taking `.parent` here points one directory
# above the project when the spec is invoked directly.
ROOT = Path(SPECPATH).resolve()
paddleocr_spec = importlib.util.find_spec('paddleocr')
if paddleocr_spec is None or not paddleocr_spec.submodule_search_locations:
    raise SystemExit('paddleocr==3.6.0 is required to build FormulaOCR.')
paddleocr_package_dir = Path(
    next(iter(paddleocr_spec.submodule_search_locations))
).resolve()
datas = [
    (str(paddleocr_package_dir), 'PaddleOCR-main/paddleocr'),
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
hiddenimports = ['paddle', 'paddlex', 'numpy', 'tokenizers', 'onnxruntime', 'rapid_latex_ocr']
datas += copy_metadata('tokenizers')
datas += copy_metadata('latex2mathml')
datas += copy_metadata('paddleocr')
for package_name in (
    'paddle',
    'paddlex',
    'cv2',
    'tokenizers',
    'pypdfium2',
    'latex2mathml',
    'onnxruntime',
    'rapid_latex_ocr',
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
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
    excludes=['tensorflow', 'torch', 'torchvision', 'torchaudio', 'modelscope', 'matplotlib', 'sklearn', 'scipy', 'paddle.tensorrt', 'paddlex.inference.serving', 'shapely.tests'],
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
