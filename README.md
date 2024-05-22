
#### Requirements
Tested combination: Python 3.9.6 + [PyTorch 1.9.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 1.7.2](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Other required python libraries include: numpy, scikit-learn, pyyaml etc.


#### Prepare Data

The location of each dataset should be
```
CODE
├── dataset
│   ├── em_user
│   ├── hpo_metab
│   ├── ppi_bp
└── └── hpo_neuro
```
#### Reproduce SubgraphMamba

To reproduce our results on real-world datasets:

```
python Subgraph-MambaTest.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device $gpu_id --dataset $dataset
```
where $dataset can be selected from em_user, ppi_bp, hpo_metab, and hpo_neuro.


#### Use Your Own Dataset

Please add a branch in the `load_dataset` function in datasets.py to load your dataset and create a configuration file in ./config to describe the hyperparameters for the Subgraph-Mamb model.
