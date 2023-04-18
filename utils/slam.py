import csv
import numpy as np

def read_transfo(path, skip_first=True):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=" ")
    i = 0
    rot = []
    trans = []
    for row in read_tsv:
        if i or not skip_first:
            row = [float(r) for r in row[:12]]
            local_rot = np.array([[row[0],row[1],row[2]],[row[4],row[5],row[6]],[row[8],row[9],row[10]]])
            local_trans = np.array([row[3],row[7],row[11]])
            rot.append(local_rot.T)
            trans.append(local_trans)
            skip_first = False
        else:
            i+=1
    return rot, trans

def apply_transformation(current_pc, transformation):
    (rot, trans) = transformation
    return np.hstack((np.dot(current_pc[:,:3],rot) + trans,current_pc[:,3:]))