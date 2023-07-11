from datasets.base_dataset import PointCloudDataset
import os.path as osp 
import os
import numpy as np 
from utils.slam import *

#The labels must be between 0 and n_class-1, -1 being the ignored value

class SemanticPOSS(PointCloudDataset):

    def __init__(self, config, split, dynamic=[3,4,5,6,7,8,9,10,12]):
        self.split = split
        self.config = config
        learning_map = config.learning_map
        self.map = np.zeros(max(learning_map.keys())+1)
        for key in learning_map:
            self.map[key] = learning_map[key]
        self.sequence = []
        self.sequence.extend(config.split[self.split])
        self.sequence = [str(s).zfill(2) for s in self.sequence]
        self.path = config.data.path
        self.dynamic = np.array(dynamic)
        self.traj_folder = config.data.traj_folder
        self.sequence_dic = {}
        for k in range(len(self.sequence)):
            self.sequence_dic[self.sequence[k]]= self.get_sequence(k)

    def loader(self, seq, frame):
        seq_path = osp.join(osp.join(self.path,'dataset/sequences'),str(seq).zfill(2))
        pointcloud = osp.join(seq_path,'velodyne',self.sequence_dic[seq][frame])
        label = osp.join(seq_path,'labels',self.sequence_dic[seq][frame].replace('.bin','.label'))
        labels_read = np.fromfile(label, dtype=np.uint32)
        sem_label = (labels_read & 0xFFFF)
        pointcloud = np.fromfile(pointcloud, dtype=np.float32, count=-1).reshape((-1, 4))
        sem_label = sem_label.astype(np.int32)
        return pointcloud, self.map_to_eval(sem_label)

    def map_to_eval(self, og_labels):
        #labels between -1 and n_label-1
        return self.map[og_labels].astype(np.int32) - 1

    def get_n_label(self):
        return int(np.max(self.map))

    def get_dynamic(self, labels):
        belonging = np.zeros_like(labels)
        for id in self.dynamic:
            belonging = np.logical_or(belonging,labels==id)
        return belonging

    def get_static(self, labels):
        return np.logical_and(np.logical_not(self.get_dynamic(labels)),labels>-1)

    def get_sequence(self,seq_number):
        return sorted(list(os.listdir(osp.join(self.path,'dataset/sequences',self.sequence[seq_number],'velodyne'))))

    def get_size_seq(self, seq_number):
        return len(os.listdir(osp.join(self.path,'dataset/sequences',self.sequence[seq_number],'velodyne')))

    def get_poses_seq(self, seq_number):
        return read_transfo(osp.join(osp.join(self.path,self.traj_folder),self.sequence[seq_number]+'_traj_complete_result.txt'),True)