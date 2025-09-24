import json
import pickle
from pathlib import Path
from typing import Any, Dict

def save_json(data: Dict[str, Any], file_path: str, indent: int = 2):
    """Save data to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_pickle(data: Any, file_path: str):
    """Save data to pickle file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: str) -> Any:
    """Load data from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
