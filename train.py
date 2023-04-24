import importlib
import argparse
from random import shuffle
from omegaconf import OmegaConf
import os.path as osp
from datasets.cluster_dataset import *
from datasets import *
import torch 
from trainer.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', help='the path to the setup config file', default='cfg/train_sk.yaml')
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
cluster_cfg = OmegaConf.load(cfg.cluster_cfg)
model_cfg = OmegaConf.load(cfg.model_cfg)
cfg = OmegaConf.merge(cfg,cluster_cfg,model_cfg)


if __name__ == "__main__":

    #Get info relative to the set. At the moment, val is required to be the same as train
    if cfg.source == "semantickitti":
        source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"semantic-kitti.yaml"))
        train_set = SemanticKITTI(source_data_cfg,'train')
        val_set = SemanticKITTI(source_data_cfg,'valid')
    elif cfg.source == "nuscenes":
        source_data_cfg = OmegaConf.load(osp.join(cfg.data_cfg_path,"nuscenes.yaml"))
        train_set = nuScenes(source_data_cfg,'train')
        val_set = nuScenes(source_data_cfg,'valid')
    else:
        raise  NameError('source dataset not supported')
    #Get info relative to the model
    if cfg.architecture.model == "KPCONV":
        module = importlib.import_module('models.kpconv.kpconv')
        model_information = getattr(module, cfg.architecture.type)()
        model_information.num_classes = train_set.get_n_label()
        model_information.ignore_label = -1
    elif cfg.architecture.model == "SPVCNN":
        module = importlib.import_module('models.spvcnn.spvcnn')
        model_information = getattr(module, cfg.architecture.type)
        model_information.num_classes = train_set.get_n_label()
        model_information.ignore_label = -1
    else:
        raise  NameError('model not supported')

    #Get info relative to the cluster set (for training)
    train_dataset = ClusterDataset(cfg,train_set)
    valid_dataset = ClusterDataset(cfg,val_set)
    tr_samplr = BalancedSampler(train_dataset.class_frames, train_set.get_n_label(),shuffle=50000, batch_size=cfg.trainer.batch_size)

    if cfg.architecture.model == "KPCONV":
        from models.kpconv_model import SemanticSegmentationModel
        module = importlib.import_module('models.kpconv.architecture')
        model_type = getattr(module, cfg.architecture.type)
        model = SemanticSegmentationModel(model_information,cfg,model_type)
    elif cfg.architecture.model == "SPVCNN":
        from models.spvcnn_model import SemanticSegmentationSPVCNNModel
        model = SemanticSegmentationSPVCNNModel(model_information,cfg)

    def collate(data):
        r_clouds, _ = model.prepare_data(data,augment=False,eval=True)
        return r_clouds
    def collate_val(data):
        r_clouds, r_inds = model.prepare_data(data,augment=False,eval=True)
        return r_clouds, data, r_inds

    #Generate dataloader
    train_dataloader = InfiniteDataLoader(
                            train_dataset,
                            batch_sampler = tr_samplr,
                            num_workers=16,
                            collate_fn=collate,
                            #batch_size=cfg.trainer.batch_size,
                            #shuffle=True
                        )
        
    valid_dataloader = torch.utils.data.DataLoader(
                            valid_dataset,
                            num_workers=16,
                            sampler=torch.utils.data.SubsetRandomSampler(np.random.choice(len(valid_dataset),cfg.trainer.evaluate_size,replace=False)),
                            collate_fn=collate_val,
                            batch_size=cfg.trainer.batch_size
                        )

    trainer = Trainer(model,train_dataloader,valid_dataloader,cfg)
    trainer.iteration_train()
