from datasets.base_dataset import PointCloudDataset
import os.path as osp 
import os
import numpy as np 
from utils.slam import *
from utils.ply import read_ply

#The labels must be between 0 and n_class-1, -1 being the ignored value

class Pandaset(PointCloudDataset):
    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.sequence = []
        self.sequence.extend(config.split[self.split])
        self.sequence = [str(s).zfill(3) for s in self.sequence]
        self.path = config.data.path

    def loader(self, seq, frame):
        ply = read_ply((os.path.join(self.path,str(seq).zfill(3),str(frame).zfill(2)+'.ply')))
        pointcloud = np.vstack((ply['x'],ply['y'],ply['z'],ply['i'])).T
        label = ply['semantic']
        return pointcloud, self.map_to_eval(label)

    def map_to_eval(self, og_labels):
        #labels between -1 and n_label-1
        return og_labels

    def get_n_label(self):
        return 42

    def get_sequence(self,seq_number):
        return list(os.listdir(osp.join(self.path,str(self.sequence[seq_number]).zfill(3)))).sort()

    def get_size_seq(self, seq_number):
        return len(os.listdir(osp.join(self.path,str(self.sequence[seq_number]).zfill(3))))

    def get_poses_seq(self, seq_number):
        return read_transfo(osp.join(osp.join(self.path,'traj_pandaset'),str(self.sequence[seq_number]).zfill(3)+'_traj_complete_result.txt'),False)