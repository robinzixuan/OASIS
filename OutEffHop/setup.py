# pyright: reportMissingImports=false, reportMissingModuleSource=false
from setuptools import setup
from pathlib import Path

scripts = [
    "run_mlm",
    "run_mlm_ddp",
    "run_clm",
    "run_clm_ddp",
    "run_clm_oasis",
    "run_vit",
    "validate_mlm",
    "validate_mlm_config",
    "validate_clm",
    "validate_vit",
]
bash_scripts = Path("scripts").glob("*.sh")

setup(
    name="attention_sinks_attention_residuals",
    version="1.0.0",
    packages=[
        "quantization",
        "quantization.quantizers",
        "transformers_language",
        "transformers_language.models",
        "vutils",
    ],
    py_modules=scripts,
    scripts=[str(path) for path in bash_scripts],
    entry_points={"console_scripts": [f"{script} = {script}:main" for script in scripts]},
    license="MIT",
    description='Implementation for "Attention Sinks and Outliers in Attention Residuals"',
)
