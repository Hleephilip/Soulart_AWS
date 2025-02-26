from lidarhuman26M_dataset import *

def build(cfgs):
    if cfgs.dataset.name == "lidarhuman26M":
        hpe_dataset = LidarHuman26MDataset()