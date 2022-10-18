import mlflow

mlflow.start_run()
mlflow.log_param("parameter name", "value")
mlflow.log_metric("metric name", 1)