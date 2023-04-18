from datasets.base_dataset import PointCloudDataset
import os.path as osp 
import os,json
import numpy as np 
from utils.slam import *

#The labels must be between 0 and n_class-1, -1 being the ignored value

class nuScenes(PointCloudDataset):
    def __init__(self, config, split, dynamic=[0,7,10,11,12,13,14,15]):
        self.split = split
        self.config = config
        learning_map = config.learning_map
        self.map = np.zeros(max(learning_map.keys())+1)
        for key in learning_map:
            self.map[key] = learning_map[key]
        self.sequence = []
        self.sequence.extend(config.split[self.split])
        self.sequence = ['scene-'+str(s).zfill(4) for s in self.sequence]
        self.path = config.data.path
        self.dynamic = np.array(dynamic)
        if os.path.exists(osp.join(self.path,'v1.0-trainval','scene2label.json')):
            with open(self.path+'/v1.0-trainval/scene2label.json', 'r') as json_file:
                self.scenes = json.load(json_file)
        else:
            self.build_scene_json()
            with open(self.path+'/v1.0-trainval/scene2label.json', 'r') as json_file:
                self.scenes = json.load(json_file)
        self.traj_folder = config.data.traj_folder

    def build_scene_json(self):
        #we are using a custom dictionnary top map sequences to their annotated frames
        from nuscenes import NuScenes

        nusc = NuScenes(version='v1.0-trainval', dataroot=self.path, verbose=True)
        scene2label = {

        }
        for sc in nusc.scene:
            scene2label[sc['name']] = {}
            file = []
            labels = []
            sample = nusc.get('sample',sc['first_sample_token'])
            sample_data = sample['data']['LIDAR_TOP']
            file.append(nusc.get_sample_data(sample_data)[0])
            labels.append(nusc.get('lidarseg',sample_data)['filename'])
            while sample['next'] != '':
                sample = nusc.get('sample',sample['next'] )
                sample_data = sample['data']['LIDAR_TOP']
                file.append(nusc.get_sample_data(sample_data)[0])
                labels.append(nusc.get('lidarseg',sample_data)['filename'])
            scene2label[sc['name']]['file'] = file
            scene2label[sc['name']]['label'] = labels

        import json 
        with open(self.path+'/v1.0-trainval/scene2label.json','w') as fp:
            json.dump(scene2label,fp)

    def loader(self, seq, idx):
        #return a tuple (pcd,labels), pcd shape = (nx4), labels shape (n)
        info = self.scene[seq]
        pointcloud = osp.join(self.path,info['file'][idx])
        pointcloud = np.fromfile(pointcloud, dtype=np.float32).reshape((-1, 5))[:, :4]
        label = osp.join(self.path,info['label'][idx])
        label = np.fromfile(label, dtype=np.uint8).astype(np.int32)

        return pointcloud, self.map_to_eval(label)

    def map_to_eval(self, og_labels):
        return self.map[og_labels].astype(np.int32) - 1

    def get_n_label(self):
        return int(np.max(self.map))

    def get_dynamic(self,labels):
        belonging = np.zeros_like(labels)
        for id in self.dynamic:
            belonging = np.logical_or(belonging,labels==id)
        return belonging

    def get_static(self,labels):
        return np.logical_and(np.logical_not(self.get_dynamic(labels)),labels>-1)


    def get_sequence(self,seq_number):
        return self.scenes[self.sequence[seq_number]]

    def get_size_seq(self, seq_number):
        return len(self.scenes[self.sequence[seq_number]]['label'])

    def get_poses_seq(self, seq_number):
        return read_transfo(osp.join(osp.join(self.path,self.traj_folder),self.sequence[seq_number]+'_traj_complete_resul.txt'),False)