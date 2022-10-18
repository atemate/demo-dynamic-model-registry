import mlflow
import joblib
import random

class Model:
    def __init__(self, value):
        self.value = value
    def predict(self, *args, **kwargs):
        return self.value


def train(i):
    experiment_name = f"train-{i}"
    experiments = mlflow.search_experiments(filter_string=f"name='{experiment_name}'", max_results=1)
    if experiments:
        experiment_id = experiments[0].experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        # train the model ...
        model = Model(i)

        # Log parameters
        mlflow.log_param("learning_rate", float(f"0.0{i}"))
        mlflow.log_param("max_depth", i+5)
        mlflow.log_param("n_estimators", i+10)

        # Log metrics
        mlflow.log_metric("auc", 1 - float(f"0.{i}"))
        mlflow.log_metric("accuracy_score", 0.9 - float(f"0.{i}"))
        mlflow.log_metric("zero_one_loss", 0.1 + float(f"0.{i}"))

        # Log artifacts
        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="my-model")



if __name__ == "__main__":
    for i in range(10):
        train(i)
    