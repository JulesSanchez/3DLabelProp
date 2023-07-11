from datasets.base_dataset import PointCloudDataset
import os.path as osp 
import os
import numpy as np 
from utils.slam import *

class SemanticKITTI(PointCloudDataset):

    def __init__(self, config, split, dynamic=[0,1,2,3,4,5,6,7]):
        self.split = split
        self.config = config
        learning_map = config.learning_map
        label_map_names = config.labels
        self.map = np.zeros(max(learning_map.keys())+1)
        self.label_names = {}
        for key in learning_map:
            self.map[key] = learning_map[key]
            if learning_map[key] not in self.label_names and learning_map[key] !=0:
                self.label_names[learning_map[key]] = label_map_names[key]
        self.label_names = list(self.label_names.values())
        self.sequence = []
        self.sequence.extend(config.split[self.split])
        self.sequence = [str(s).zfill(2) for s in self.sequence]
        self.path = config.data.path
        self.traj_folder = config.data.traj_folder
        self.dynamic = np.array(dynamic)

    def loader(self, seq, frame):
        seq_path = osp.join(osp.join(self.path,'dataset/sequences'),str(seq).zfill(2))
        pointcloud = osp.join(osp.join(seq_path,'velodyne'),str(frame).zfill(6)+'.bin')
        label = osp.join(osp.join(seq_path,'labels'),str(frame).zfill(6)+'.label')
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
        return list(os.listdir(osp.join(self.path,'dataset/sequences',self.sequence[seq_number]))).sort()

    def get_size_seq(self, seq_number):
        return len(os.listdir(osp.join(self.path,'dataset/sequences',self.sequence[seq_number],'velodyne')))

    def get_poses_seq(self, seq_number):
        return read_transfo(osp.join(osp.join(self.path,self.traj_folder),self.sequence[seq_number]+'_traj.txt'),False)