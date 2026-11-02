minio-run:
	minio server /Users/baptistedarnala/Documents/Database/minio/ --address :9500 --console-address :9501 --license ~/Documents/Database/minio/minio.license

mlflow-run:
	uv run --package model-evaluation mlflow server

export-labels:
	uv run python conversation-engine/export_labels.py

evaluate-stt:
	uv run --package model-evaluation evaluate-stt $(if $(DATASET),--dataset $(DATASET))
