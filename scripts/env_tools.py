from pathlib import Path
from dotenv import load_dotenv
import os

def getenv(key: str, default=None):
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root/".env")
    return os.getenv(key, default)
