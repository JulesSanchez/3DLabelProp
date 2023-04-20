from models.spvcnn.architecture import SPVCNN
import numpy as np 
import pickle, os
import torch
from os.path import join
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

class SemanticCustomBatch:
    def __init__(self, input_list):
        self.inputs = sparse_collate_fn(input_list)
        self.labels = self.inputs['targets'].F
        self.neigbhors = [None]

    def to(self, device):
        self.inputs['lidar'].to(device)
        self.labels.to(device)


class SemanticSegmentationSPVCNNModel:
    def __init__(self, model_config, config):
        lbl_values = [i for i in range(model_config.num_classes)]
        ign_lbls = [model_config.ignore_label]
        self.model = SPVCNN(num_classes=model_config.num_classes,
                                cr=model_config.cr,
                                pres=model_config.voxel_size,
                                vres=model_config.voxel_size,
                                in_feats=model_config.in_features_dim)
        self.model_config = model_config
        self.data_config = config
        print("Model ready")

    def prepare_data(self, pointclouds, augment=True,eval=False):
        inputs = []
        for pc in pointclouds:
            pc = pc.astype(np.float32)
            #center if asked
            merged_points = pc[:,:3]
            if self.model_config.augment_center:
                merged_points -= np.mean(merged_points, axis=0)
            if augment:
                do_rotation = np.random.uniform()
                do_scale = np.random.uniform()
                do_noise = np.random.uniform()
                if do_rotation > 0.5:
                    theta = np.random.uniform(-np.pi/2,np.pi/2)
                    r = R.from_rotvec(theta * np.array([0, 0, 1]))
                    merged_points[:,:3] = r.apply(merged_points[:,:3])
                if do_scale > 0.5:
                    scale = np.random.uniform(0.8,1.2)
                else:
                    scale = 1
                if do_noise > 0.5:
                    noise = 0.001
                    noise = (np.random.randn(merged_points.shape[0], 3) * noise).astype(np.float32)
                else:
                    noise = 0

                merged_points[:,:3] = merged_points[:,:3] * scale + noise

            merged_coords = pc[:,np.array([0,1,2,3])]
            merged_labels = pc[:,4]

            pc_ = np.round(merged_coords[:, :3] / self.model_config.voxel_size).astype(np.int32)
            _, inds, inverse_map = sparse_quantize(pc_,
                                                return_index=True,
                                                return_inverse=True)

            red_pc = pc_[inds]
            feat = np.ones_like(merged_coords[inds, :1], dtype=np.float32)
            if self.model_config.in_features_dim>1:
                #add z, then add r
                feat = np.hstack((feat, merged_coords[inds, 2:2+self.model_config.in_features_dim-1]))
            labels = merged_labels[inds]
            inputs.append({
                'lidar': SparseTensor(feats=feat, coords=red_pc),
                'targets': SparseTensor(feats=labels, coords=red_pc),
                'targets_mapped': SparseTensor(feats=merged_labels, coords=pc_),
                'inverse_map': SparseTensor(feats=inverse_map, coords=pc_),
            })
            inputs = SemanticCustomBatch(inputs)
        return inputs, inputs['inverse_map']
