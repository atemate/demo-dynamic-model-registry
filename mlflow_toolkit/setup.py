from setuptools import find_packages, setup

setup(
    name="mlflow_toolkit",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["mlflow", "sklearn", "pandas"],
)
