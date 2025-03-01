import yaml
from easydict import EasyDict as edict

def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config

def print_config(cfg, offset=""):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{offset}{k}:")
            print_config(v, offset=offset + "   ")
        else:
            print(f"{offset}{k}: {v}")