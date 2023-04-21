import importlib
import argparse
from omegaconf import OmegaConf
import os.path as osp
from datasets.inference_dataset import *
from datasets import *
import torch 

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', help='the path to the setup config file', default='cfg/train_sk.yaml')
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
cluster_cfg = OmegaConf.load(cfg.cluster_cfg)
model_cfg = OmegaConf.load(cfg.model_cfg)
cfg = OmegaConf.merge(cfg,cluster_cfg,model_cfg)
if __name__ == "__main":
    #Get info relative to the set
    if cfg.source == "semantickitti":
        source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
        train_set = SemanticKITTI(source_data_cfg,'valid')
    elif cfg.source == "nuscenes":
        source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes.yaml"))
        train_set = nuScenes(source_data_cfg,'valid')
    else:
        raise  NameError('source dataset not supported')

    if cfg.target == "semantickitti":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
        val_set = SemanticKITTI(target_data_cfg,'valid')
    elif cfg.target == "nuscenes":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes.yaml"))
        val_set = nuScenes(target_data_cfg,'valid')
    elif cfg.target == "semanticposs":
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semanticposs.yaml"))
        val_set = SemanticPOSS(target_data_cfg,'valid')
    elif "pandaset" in cfg.target:
        target_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,cfg.target+".yaml"))
        val_set = Pandaset(target_data_cfg,'valid')
    else:
        raise  NameError('target dataset not supported')

    #Get info relative to the model
    if cfg.architecture.model == "KPCONV":
        module = importlib.import_module('models.kpconv.kpconv')
        model_information = getattr(module, cfg.architecture.type)()
        model_information.num_classes = train_set.get_n_label()
        model_information.ignore_label = -1
        from models.kpconv_model import SemanticSegmentationModel
        module = importlib.import_module('models.kpconv.architecture')
        model_type = getattr(module, cfg.architecture.type)
        model = SemanticSegmentationModel(model_information,cfg,model_type)
    elif cfg.architecture.model == "SPVCNN":
        module = importlib.import_module('models.spvcnn.spvcnn')
        model_information = getattr(module, cfg.architecture.type)
        model_information.num_classes = train_set.get_n_label()
        model_information.ignore_label = -1
        from models.spvcnn_model import SemanticSegmentationSPVCNNModel
        model = SemanticSegmentationSPVCNNModel(model_information,cfg)
    else:
        raise  NameError('model not supported')

    valid_dataset = InferenceDataset(cfg,train_set,val_set,model,model_information)
    try:
        ius, miu = valid_dataset.compute_results()
    except:
        valid_dataset.compute_dataset()
        ius, miu = valid_dataset.compute_results()
    print(ius)
    print(miu)

