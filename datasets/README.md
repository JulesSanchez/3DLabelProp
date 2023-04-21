# Dataset

## Supported datasets

At the moment for training, there is support for nuScenes and SemanticKITTI. For inference: nuScenes, SemanticKITTI, Pandaset, SemanticPOSS.

## Add a new dataset

To add a new dataset, several step must be taken. The dataset needs to be accompanied with the trajectories for each sequence. 

* Add {name_of_the_dataset}.py file, and create the class following the PointCloudDataset signature. If your trajectories have a header line, put skip_first=True in in the read_transfo function.
* Add the new class and necessary import to the \__init__.py file
* Add {name_of_the_dataset}.yaml file, which should include the path of the data, the train/val/test split (if it is a dataset for evaluation, put everything in val), and the information necessary for your Class relative to the dataset (labels, mapping, etc).
* Add the necessary support for this new config file in train.py and in infer.py. If it is only for eval, include it in infer.py in the target case.
* If it will be used for training, in ./mapping, add an adequate file which will be used for evaluation. Add the import in inference-dataset.py, function compute_results.
* If it will be used for evaluation, in ./mapping/sk.py and in ./mapping/ns.py, in the dictionnaries add a new key with the name of the dataset and the mapping used for evaluation. This new added dictionnary must have an entry "labels_name", "source_to_common", "target_to_common".