import numpy as np 

#For training sets, the labels must be between 0 and n_class-1, -1 being the ignored value

class PointCloudDataset:
    def __init__(self, config):
        pass 

    def loader(self, seq, idx):
        #return a tuple (pcd,labels), pcd shape = (nx4), labels shape (n)
        pass

    def map_to_eval(self, og_labels):
        pass

    def get_n_label(self):
        pass

    def get_dynamic(self):
        pass 

    def get_static(self):
        pass

    def get_sequences(self):
        pass 

    def get_size_seq(self):
        pass

    def get_class_names(self):
        pass