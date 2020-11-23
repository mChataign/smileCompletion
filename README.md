# smileCompletion
Completion of financial dataset.

Experiments were produced on multiple notebooks that are compatible with tensorflow (TensorflowNowcasting.ipynnb works with tensorflow version >= 1.15) or pytorch (pytorchNowcasting.ipynb is compatible for torch version >= 1.6.0).
Results presented in "Nowcasting networks" paper were obtained with pytorchNowcasting notebook.

We provide data for equity volatilities in folder "Data/SPX".

"Code" folder contains implementation of autoencoders, functional approach and interpolation benchmark in python scripts.
Another README.md at the root of "Code" folder details the role of each python script.

"adam_api_repo_curve_anomaly_detection" contains data, dependancies and codes for the outlier detection on repo curves.
See in particular outlier_run_notebook.ipynb and outlier_train.ipynb.




