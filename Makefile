minio-run:
	minio server /Users/baptistedarnala/Documents/Database/minio/ --address :9500 --console-address :9501 --license ~/Documents/Database/minio/minio.license

export-labels:
	uv run python conversation-engine/export_labels.py
