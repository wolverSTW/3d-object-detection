import yaml
from pathlib import Path


def load_yaml_config(path: str):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with config_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)
