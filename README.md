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

-install torchsparse https://github.com/mit-han-lab/torchsparse
-install pytorch, here with conda
`conda create --name pytorch python=3.7`
`conda activate pytorch`
`conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch`
-install other requirements
`pip install -r requirements.txt`

### Compilation of C++

`cd cpp_wrappers`
`bash compile_wrappers.sh`

## Usage

### Train
Be careful, for training, we preprocess the dataset, for SemanticKITTI with default parameters, it takes approximately 200Go.
Change the paths appropriately in the various config files.
Traj files can be found in: XXX. Put the unzipped folder at the root of the dataset.

By default, train a model on semantickitti with KPConv.

`train.py`

### Inference
Be careful, by default, inferences as saved on disk for future use.
Change the paths appropriately in the various config files.

By default, infer a model trained on semantickitti with KPConv on semantickitti

`infer.py`

## Trained models
Trained models on semantickitti and nuscenes with KPConv can be found at XXX.

## Credits

Thanks for the original authors of KPConv (https://github.com/HuguesTHOMAS/KPConv-PyTorch) and SPVCNN (https://github.com/mit-han-lab/spvnas) from which we 
If you use this repo please cite us:
@misc{sanchez2023domain,
      title={Domain generalization of 3D semantic segmentation in autonomous driving}, 
      author={Jules Sanchez and Jean-Emmanuel Deschaud and Francois Goulette},
      year={2023},
      eprint={2212.04245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}