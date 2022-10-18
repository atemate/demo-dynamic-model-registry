from pathlib import Path

from setuptools import find_packages, setup

REQUIREMENTS_TXT = Path(__file__).parent / "requirements.txt"
REQUIREMENTS = REQUIREMENTS_TXT.read_text().splitlines()


setup(
    name="mlflow_toolkit",
    version="0.0.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
