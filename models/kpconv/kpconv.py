#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Configuration class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


from os.path import join
import numpy as np


# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # Dataset name
    dataset = ''

    # Type of network model
    dataset_task = ''

    # Number of classes in the dataset
    num_classes = 0

    # Dimension of input points
    in_points_dim = 3

    # Dimension of input features
    in_features_dim = 1

    # Radius of the input sphere (ignored for models, only used for point clouds)
    in_radius = 4.0

    # Number of CPU threads for the input pipeline
    input_threads = 8

    ##################
    # Model parameters
    ##################

    # Architecture definition. List of blocks
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # Decide the mode of equivariance and invariance
    equivar_mode = ''
    invar_mode = ''

    # Dimension of the first feature maps
    first_features_dim = 128

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.99

    # For segmentation models : ratio between the segmented area and the input area
    segmentation_ratio = 1.0

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Kernel point influence radius
    KP_extent = 1.2

    # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    # Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
    aggregation_mode = 'sum'

    # Fixed points in the kernel : 'none', 'center' or 'verticals'
    fixed_kernel_points = 'center'

    # Use modulateion in deformable convolutions
    modulated = False

    # For SLAM datasets like SemanticKitti number of frames used (minimum one)
    n_frames = 1

    # For SLAM datasets like SemanticKitti max number of point in input cloud + validation
    max_in_points = 100000
    val_radius = 4.0
    max_val_points = 100000

    #####################
    # Training parameters
    #####################

    # Network optimizer parameters (learning rate and momentum)
    learning_rate = 1e-2
    momentum = 0.98

    # Learning rate decays. Dictionary of all decay values with their epoch {epoch: decay}.
    lr_decays = {200: 0.2, 300: 0.2}

    # Gradient clipping value (negative means no clipping)
    grad_clip_norm = 100.0

    # Augmentation parameters
    augment_scale_anisotropic = True
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_noise = 0.001
    augment_color = 0.7
    augment_center = True

    # Augment with occlusions (not implemented yet)
    augment_occlusion = 'none'
    augment_occlusion_ratio = 0.2
    augment_occlusion_num = 1

    # Regularization loss importance
    weight_decay = 1e-3

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    class_w = []
    class_threshold = 30

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    # Number of batch
    batch_num = 1
    val_batch_num = 1
    batch_size = 1

    # Do we nee to save convergence
    saving = True
    saving_path = None

    def __init__(self):
        """
        Class Initialyser
        """

        # Number of layers
        self.num_layers = len([block for block in self.architecture if 'pool' in block or 'strided' in block]) + 1

        ###################
        # Deform layer list
        ###################
        #
        # List of boolean indicating which layer has a deformable convolution
        #

        layer_blocks = []
        self.deform_layers = []
        arch = self.architecture
        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    deform_layer = True

            if 'pool' in block or 'strided' in block:
                if 'deformable' in block:
                    deform_layer = True

            self.deform_layers += [deform_layer]
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

    def load(self, path):

        filename = join(path, 'parameters.txt')
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Class variable dictionary
        for line in lines:
            line_info = line.split()
            if len(line_info) > 2 and line_info[0] != '#':

                if line_info[2] == 'None':
                    setattr(self, line_info[0], None)

                elif line_info[0] == 'lr_decay_epochs':
                    self.lr_decays = {int(b.split(':')[0]): float(b.split(':')[1]) for b in line_info[2:]}

                elif line_info[0] == 'architecture':
                    self.architecture = [b for b in line_info[2:]]

                elif line_info[0] == 'augment_symmetries':
                    self.augment_symmetries = [bool(int(b)) for b in line_info[2:]]

                elif line_info[0] == 'num_classes':
                    if len(line_info) > 3:
                        self.num_classes = [int(c) for c in line_info[2:]]
                    else:
                        self.num_classes = int(line_info[2])

                elif line_info[0] == 'class_w':
                    self.class_w = [float(w) for w in line_info[2:]]

                elif hasattr(self, line_info[0]):
                    attr_type = type(getattr(self, line_info[0]))
                    if attr_type == bool:
                        setattr(self, line_info[0], attr_type(int(line_info[2])))
                    else:
                        setattr(self, line_info[0], attr_type(line_info[2]))

        self.saving = True
        self.saving_path = path
        self.__init__()


class KPFCNN(Config):
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    arch_type = 'default'
    in_radius = 4.0
    val_radius = 4.0
    # KPConv parameters
    max_in_points = 30000
    # Number of kernel points
    num_kernel_points = 15
    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06
    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5
    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0
    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2
    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'
    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'
    # Choice of input features
    first_features_dim = 128
    # Dimension of input features
    in_features_dim = 1