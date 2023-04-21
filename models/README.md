# Models

## Supported models

At the moment, the supported models are SPVCNN and KPConv. Adding a new backbone is a bit tedious, apologies.

## Add a new model

To add a new model:
* Create a new file {model_name}_model.py. Add a foler {model_name}. In the folder will be contained all the modules necessary to define your new model.
* In the {model_name}_model.py, create a model class. The class must incorporate a .model element which can be called for inference. We recommend including a .prepare_data method which process a list of point clouds and output the well formated batch taht can be fed to .model.
* Add the necessary import and loading in train.py and infer.py.
* Add the encessary colalte_function in train.py.
* if it requires a specific optimizer, add the support in trainer/trainer.py
* In trainer/trainer.py, add an .evaluate_{model_name} method and add the call in iteration train.
* In datasets/inference_dataset.py, add and infer_concat_{model name} function which can run the inference on a list of datasets and output the per point predictions (softamx output, not argmax). And call it in in \__init__.py of InferenceDataset.