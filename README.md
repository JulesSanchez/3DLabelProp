# 3DLabelProp

## Installation

### C++ requirements

- mlpack (https://www.mlpack.org/)
- openmp
- pybind11 (https://github.com/pybind/pybind11)
- armadillo (https://arma.sourceforge.net/)

### Cuda requirements

This code was tested with CUDA 11.1

### Python requirements

- install torchsparse https://github.com/mit-han-lab/torchsparse
- install pytorch, here with conda

`conda create --name pytorch python=3.7`

`conda activate pytorch`

`conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch`
- install other requirements

`pip install -r requirements.txt`

### Compilation of C++

`cd cpp_wrappers`

`bash compile_wrappers.sh`

## Usage

### Train
Be careful, for training, we preprocess the dataset. For example, for SemanticKITTI with the default parameters, you need approximately 200Go of free storage.

Change the paths appropriately in the various config files (./cfg).

Trajectory files can be found in: XXX. 

Put the unzipped folder at the root of the dataset.

By default, to train a model on SemanticKITTI with KPConv use:

`python train.py`

### Inference
Be careful, by default, inferences as saved on disk for future use.

Change the paths appropriately in the various config files.

By default, to infer a model trained on SemanticKITTI with KPConv on SemanticKITTI use:

`python infer.py`

## Trained models
Trained models on SemanticKITTI and nuScenes with KPConv can be found at XXX.

## Credits

Thanks to the original authors of KPConv (https://github.com/HuguesTHOMAS/KPConv-PyTorch) from which we copied the KPConv backbone code and SPVCNN (https://github.com/mit-han-lab/spvnas) from which we copied the SPVCNN backbone code.

If you use this repo please cite us:

@misc{sanchez2023domain,
      title={Domain generalization of 3D semantic segmentation in autonomous driving}, 
      author={Jules Sanchez and Jean-Emmanuel Deschaud and Francois Goulette},
      year={2023},
      eprint={2212.04245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}