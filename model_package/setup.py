from setuptools import find_packages, setup

setup(
    name="mlflow_toolkit",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "mlflow==1.29.0",
        "sklearn",  # for mlflow.sklearn.log_model
        "joblib",
    ],
)
