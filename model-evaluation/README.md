# Model evaluation

Compares the models of the assistant and logs the results in MLflow. Each type of model has its own folder and MLflow experiment.

- `stt/`: runs the Whisper models on the labeled recordings of the bucket, with `make evaluate-stt`.

Start the MLflow server first, with `make mlflow-run`.
