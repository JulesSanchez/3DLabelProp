import importlib
import argparse
from omegaconf import OmegaConf
import os.path as osp

target_data_cfg = OmegaConf.load(osp.join('cfg/data_cfg',"semantic-kitti.yaml"))

print(target_data_cfg.learning_map)