from dotenv import load_dotenv

load_dotenv()

from conversation_engine.storage.labels import export_labels

if __name__ == "__main__":
    print(f"{export_labels()} label(s) exported")
