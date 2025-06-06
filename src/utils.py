import hashlib
import json
from pathlib import Path
from types import ModuleType


def load_hyperparameters(hp_path: Path) -> dict:
    """Load hyperparameters from a file and compute its hash."""
    with open(hp_path, "r") as f:
        hparams_str = f.read()
    hparams_hash = hashlib.shake_128(hparams_str.encode()).hexdigest(3)
    hparams = json.loads(hparams_str)
    return hparams, hparams_hash

 

def init_from_config(config: dict, module: ModuleType, *args, **kwargs):
    return getattr(module, config["name"])(*args, **kwargs, **config.get("params", {}))
