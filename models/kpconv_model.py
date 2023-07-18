from models.kpconv.architecture import KPFCNN
from models.kpconv.common import PointCloudDataset, grid_subsampling
import numpy as np 
import pickle, os
import torch
from os.path import join
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R


class SemanticCustomBatch:
    def __init__(self, input_list):
        # Get rid of batch dimension
        # Number of layers
        if isinstance(input_list, dict):
            self.points = input_list['points']
            self.neighbors = input_list['neighbors']
            self.pools = input_list['pools']
            self.upsamples = input_list['upsamples']
            self.lengths = input_list['lengths']
            self.features = input_list['features']
            self.labels = input_list['labels']

        else:
            L = int(input_list[0])
            ind = 1
            self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
            ind += L
            self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
            ind += L
            self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
            ind += L
            self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
            ind += L
            self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
            ind += L
            self.features = torch.from_numpy(input_list[ind])
            ind += 1
            self.labels = torch.from_numpy(input_list[ind])
            ind += 1
            self.timestamps = torch.from_numpy(input_list[ind])
        return    
    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.timestamps = self.timestamps.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list

class SemanticSegmentationModel:
    def __init__(self, model_config, config, model=None):
        lbl_values = [i for i in range(model_config.num_classes)]
        ign_lbls = [model_config.ignore_label]
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:
            dev = "cpu"
        self.device = torch.device(dev) 
        if model == None:
            self.model = KPFCNN(model_config, lbl_values, ign_lbls) 
        else:
            self.model = model(model_config, lbl_values, ign_lbls) 
        self.model.to(self.device)
        self.model_config = model_config
        self.config = config
        self.helper_function = PointCloudDataset('token', self.model_config)
        self.helper_function.config = self.model_config
        self.path = join(self.config.cluster.path,self.config.source,self.config.cluster.name)
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        os.makedirs(join(self.path,'kpconv_files'),exist_ok=True)
        try:
            self.calibration()
        except:
            self.compute_calibration()
            self.calibration()
        print("Model ready")

    def compute_calibration(self,untouched_ratio=0.8):
        max_in_lim_file = join(self.path, 'kpconv_files/max_in_limits' + self.model_config.arch_type + '.pkl')
        max_in_lim_dict = {}
        key = '{:s}_{:.3f}_{:.3f}'.format('random',
                                          self.model_config.max_in_points,
                                          self.model_config.first_subsampling_dl)
        i = 0
        breaking = False
        all_lengths = []
        N = 1000
        for folder in os.listdir(self.path):
            if 'kp' not in folder and '.yaml' not in folder and "train" not in folder and "valid" not in folder:
                seq_path = join(self.path,folder)
                frame_list = os.listdir(seq_path)
                for k in range(1,len(frame_list)-1,10):
                    clusters = np.fromfile(join(self.path,folder,frame_list[k]),dtype=np.float32).reshape(-1,6)
                    batch, r_inds_list = self.prepare_data([clusters])
                    all_lengths += batch.lengths[0].tolist()
                    if len(all_lengths) > N:
                        breaking = True
                        break


        max_in_lim_dict[key] = int(np.percentile(all_lengths, 100*untouched_ratio))
        with open(max_in_lim_file, 'wb') as file:
            print('dumped in ', max_in_lim_file)
            pickle.dump(max_in_lim_dict, file)

        batch_lim_file = join(self.path, 'kpconv_files/batch_limits' + self.model_config.arch_type + '.pkl')
        neighb_lim_file = join(self.path, 'kpconv_files/neighbors_limits' + self.model_config.arch_type + '.pkl')
        batch_lim_dict = {}
        neighb_lim_dict = {}
        key = '{:s}_{:.3f}_{:.3f}_{:d}_{:d}'.format('random',
                                                    self.model_config.in_radius,
                                                    self.model_config.first_subsampling_dl,
                                                    self.model_config.batch_num,
                                                    self.model_config.max_in_points)

        ############################
        # Neighbors calib parameters
        ############################

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (self.model_config.deform_radius + 1) ** 3))

        # Histogram of neighborhood sizes
        neighb_hists = np.zeros((self.model_config.num_layers, hist_n), dtype=np.int32)

        ########################
        # Batch calib parameters
        ########################

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.model_config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        # Save input pointcloud sizes to control max_in_points
        cropped_n = 0
        all_n = 0

        # Loop parameters
        i = 0
        breaking = False

        #####################
        # Perform calibration
        #####################

        #self.dataset.batch_limit[0] = self.dataset.max_in_points * (self.dataset.batch_num - 1)
        for folder in os.listdir(self.path):
            if 'kp' not in folder and '.yaml' not in folder and "train" not in folder and "valid" not in folder:
                seq_path = join(self.path,folder)
                frame_list = list(filter(lambda el: el.endswith('.bin'), list(os.listdir(seq_path))))
                for k in range(1,len(frame_list)-1,10):
                    clusters = np.fromfile(join(self.path,folder,frame_list[k]),dtype=np.float32).reshape(-1,6)
                    batch, r_inds_list = self.prepare_data([clusters])
                    # Control max_in_points value
                    are_cropped = batch.lengths[0] > self.model_config.max_in_points - 1
                    cropped_n += torch.sum(are_cropped.type(torch.int32)).item()
                    all_n += int(batch.lengths[0].shape[0])

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)
                    i += 1

        # Use collected neighbor histogram to get neighbors limit
        cumsum = np.cumsum(neighb_hists.T, axis=0)
        percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
        self.helper_function.neighborhood_limits = percentiles

        # Save neighb_limit dictionary
        for layer_ind in range(self.model_config.num_layers):
            dl = self.model_config.first_subsampling_dl * (2 ** layer_ind)
            if self.model_config.deform_layers[layer_ind]:
                r = dl * self.model_config.deform_radius
            else:
                r = dl * self.model_config.conv_radius
            key = '{:s}_{:d}_{:.3f}_{:.3f}'.format('random', self.model_config.max_in_points, dl, r)
            neighb_lim_dict[key] = self.helper_function.neighborhood_limits[layer_ind]
        with open(neighb_lim_file, 'wb') as file:
            pickle.dump(neighb_lim_dict, file)
            print('dumped in ', neighb_lim_file)

    def calibration(self):
        #It assumes all precomputation were done

        neighb_lim_file = join(self.path, 'kpconv_files/neighbors_limits' + self.model_config.arch_type + '.pkl')
        with open(neighb_lim_file, 'rb') as file:
            neighb_lim_dict = pickle.load(file)

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.model_config.num_layers):

            dl = self.model_config.first_subsampling_dl * (2**layer_ind)
            if self.model_config.deform_layers[layer_ind]:
                r = dl * self.model_config.deform_radius
            else:
                r = dl * self.model_config.conv_radius

            key = '{:s}_{:d}_{:.3f}_{:.3f}'.format('random', self.model_config.max_in_points, dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]
         
        self.helper_function.neighborhood_limits = neighb_limits

    def batch(self,input_list):
        return SemanticCustomBatch(input_list)

    def to(self, device):
        self.model.to(device)

    def prepare_data(self, pointclouds, augment=True,eval=False):
        #size = [len(sub_cloud) for sub_cloud in pointclouds]
        size = []
        p_list = []
        f_list = []
        l_list = []
        r_inds_list = []
        r_mask_list = []
        t_list = []
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

            merged_coords = pc[:,np.array([0,1,2,3,-1])]
            merged_labels = pc[:,4]

            in_pts, in_fts, in_lbls = grid_subsampling(merged_points,
                                                        features=merged_coords,
                                                        labels=merged_labels.astype(np.int32),
                                                        sampleDl=self.model_config.first_subsampling_dl)
            p_list += [in_pts]
            f_list += [in_fts]
            l_list += [in_lbls.reshape(-1)]
            t_list += [in_fts[:,-1].reshape(-1)]
            # Project predictions on the frame points
            if eval:
                search_tree = KDTree(in_pts, leaf_size=50)
                proj_inds = search_tree.query(pc[:,:3], return_distance=False)
                proj_inds = np.squeeze(proj_inds).astype(np.int32)
            else:
                proj_inds = None

            r_inds_list += [proj_inds]
            size.append(len(in_pts))
        stacked_points = np.concatenate(p_list, axis=0)
        timestamps = np.concatenate(t_list, axis =0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        features = features.astype(np.float32)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.model_config.in_features_dim==2:
            #add ts
            where = features[:,4] == np.max(features[:,4])
            features[:,4] = - 1
            features[where,4] = 1
            stacked_features = np.hstack((stacked_features, features[:, -1].reshape(-1,1)))
        elif self.model_config.in_features_dim==3:
            #add r and ts
            do_r = np.random.uniform()
            if do_r > 0.5:
                features[:,3] = 1
            where = features[:,4] == np.max(features[:,4])
            features[:,4] = - 1
            features[where,4] = 1
            stacked_features = features[:, np.array([3,4])]
        input_list = self.helper_function.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels.astype(np.int64),
                                              np.array(size, dtype=np.int32))

        return self.batch([self.model_config.num_layers] + input_list + [timestamps]), r_inds_list