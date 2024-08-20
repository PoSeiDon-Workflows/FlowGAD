# Graph Neural Network for Anomaly Detection and Classification in Scientific Workflows

## Repo Content

- **adjacency_list_dags**: json files with dependencies between nodes in each workflow
- **data**: raw data - characterizations of jobs in each workflow during multiple execustions
- **data_new**: raw data - new workflows with individual job anomalies
- **deepHyp_scripts**: scripts connected to finding best hyperparameters for GNN
- **psd_gnn**: module folder
<!-- - **results**: csv files with results -->
<!-- - **helpers**: helper scripts that include functions and class definitions (models) -->
<!-- - **notebooks**: notebooks used during the code development -->
<!-- - **pickles**: pickle data from previous experiments (processsed raw data) -->
<!-- - **preprocess_graph_data**: pickles with workflow data in format expected by PyG -->
<!-- - **submission_scripts**: bash scripts for jobs submissions -->

## setup env

* Install with CUDA available (default)
  
  `sh setup.sh gpu`
  
* Install with CPU only

  `sh setup.sh cpu`

### Step 0 : ETA

- visualize the graphs
- feature analysis

### Data Description

<!-- ```python3
python3 preprocess_data.py --workflow_type all
``` -->

graphs stat
|                        | cpu samples | hdd samples | loss samples | normal samples | # of nodes | # of edges |
| :--------------------- | :---------: | :---------: | :----------: | :------------: | ---------: | ---------: |
| 1000gemome             |     250     |     575     |     250      |      200       |         57 |        129 |
| nowcast-clustering-8   |     300     |     360     |     300      |      270       |         13 |         20 |
| nowcast-clustering-16  |     300     |     360     |     300      |      270       |          9 |         12 |
| wind-clustering-casa   |     300     |     360     |     300      |      270       |          7 |          8 |
| wind-noclustering-casa |     300     |     360     |     300      |      270       |         26 |         44 |


### Running scripts

We have two main scripts to run the experiments under `examples` folder. The scripts are:

* graph-level anomaly detection
```
$ python demo_graph_classification.py --help
usage: demo_graph_classification.py [-h]
                                    [--workflow {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,montage,predict_future_sales,casa-wind-full,all}]
                                    [--binary] [--gpu GPU] [--epoch EPOCH] [--hidden_size HIDDEN_SIZE] [--batch_size BATCH_SIZE] [--conv_blocks CONV_BLOCKS] [--train_size TRAIN_SIZE]
                                    [--lr LR] [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT] [--feature_option FEATURE_OPTION] [--seed SEED] [--path PATH] [--log] [--logdir LOGDIR]
                                    [--force] [--balance] [--verbose] [--output] [--anomaly_cat ANOMALY_CAT] [--anomaly_level [ANOMALY_LEVEL ...]] [--anomaly_num ANOMALY_NUM]

options:
  -h, --help            show this help message and exit
  --workflow {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,montage,predict_future_sales,casa-wind-full,all}, -w {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,montage,predict_future_sales,casa-wind-full,all}
                        Name of workflow.
  --binary              Toggle binary classification.
  --gpu GPU             GPU id. `-1` for CPU only.
  --epoch EPOCH         Number of epoch in training.
  --hidden_size HIDDEN_SIZE
                        Hidden channel size.
  --batch_size BATCH_SIZE
                        Batch size.
  --conv_blocks CONV_BLOCKS
                        Number of convolutional blocks
  --train_size TRAIN_SIZE
                        Train size [0.5, 1). And equal split on validation and testing.
  --lr LR               Learning rate.
  --weight_decay WEIGHT_DECAY
                        Weight decay for Adam.
  --dropout DROPOUT     Dropout in neural networks.
  --feature_option FEATURE_OPTION
                        Feature option.
  --seed SEED           Fix the random seed. `-1` for no random seed.
  --path PATH, -p PATH  Specify the root path of file.
  --log                 Toggle to log the training
  --logdir LOGDIR       Specify the log directory.
  --force               To force reprocess datasets.
  --balance             Enforce the weighted loss function.
  --verbose, -v         Toggle for verbose output.
  --output, -o          Toggle for pickle output file.
  --anomaly_cat ANOMALY_CAT
                        Specify the anomaly set.
  --anomaly_level [ANOMALY_LEVEL ...]
                        Specify the anomaly levels. Multiple inputs.
  --anomaly_num ANOMALY_NUM
                        Specify the anomaly num from nodes.
```

* node-level anomaly detection
```
$ python demo_node_classification.py --help 
usage: demo_node_classification.py [-h]
                                   [--workflow {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,montage,predict_future_sales,casa-wind-full,all}]
                                   [--binary] [--gpu GPU] [--epoch EPOCH] [--hidden_size HIDDEN_SIZE] [--batch_size BATCH_SIZE] [--conv_blocks CONV_BLOCKS] [--train_size TRAIN_SIZE]
                                   [--lr LR] [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT] [--feature_option FEATURE_OPTION] [--seed SEED] [--path PATH] [--log] [--logdir LOGDIR]
                                   [--force] [--balance] [--verbose] [--output] [--anomaly_cat ANOMALY_CAT] [--anomaly_level [ANOMALY_LEVEL ...]] [--anomaly_num ANOMALY_NUM]

options:
  -h, --help            show this help message and exit
  --workflow {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,montage,predict_future_sales,casa-wind-full,all}, -w {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,montage,predict_future_sales,casa-wind-full,all}
                        Name of workflow.
  --binary              Toggle binary classification.
  --gpu GPU             GPU id. `-1` for CPU only.
  --epoch EPOCH         Number of epoch in training.
  --hidden_size HIDDEN_SIZE
                        Hidden channel size.
  --batch_size BATCH_SIZE
                        Batch size.
  --conv_blocks CONV_BLOCKS
                        Number of convolutional blocks
  --train_size TRAIN_SIZE
                        Train size [0.5, 1). And equal split on validation and testing.
  --lr LR               Learning rate.
  --weight_decay WEIGHT_DECAY
                        Weight decay for Adam.
  --dropout DROPOUT     Dropout in neural networks.
  --feature_option FEATURE_OPTION
                        Feature option.
  --seed SEED           Fix the random seed. `-1` for no random seed.
  --path PATH, -p PATH  Specify the root path of file.
  --log                 Toggle to log the training
  --logdir LOGDIR       Specify the log directory.
  --force               To force reprocess datasets.
  --balance             Enforce the weighted loss function.
  --verbose, -v         Toggle for verbose output.
  --output, -o          Toggle for pickle output file.
  --anomaly_cat ANOMALY_CAT
                        Specify the anomaly set.
  --anomaly_level [ANOMALY_LEVEL ...]
                        Specify the anomaly levels. Multiple inputs.
  --anomaly_num ANOMALY_NUM
                        Specify the anomaly num from nodes.
```

## Reference


```
@inproceedings{jin2022workflow,
  title={Workflow anomaly detection with graph neural networks},
  author={Jin, Hongwei and Raghavan, Krishnan and Papadimitriou, George and Wang, Cong and Mandal, Anirban and Krawczuk, Patrycja and Pottier, Lo{\"\i}c and Kiran, Mariam and Deelman, Ewa and Balaprakash, Prasanna},
  booktitle={2022 IEEE/ACM Workshop on Workflows in Support of Large-Scale Science (WORKS)},
  pages={35--42},
  year={2022},
  organization={IEEE}
}

@article{jin2023graph,
  title={Graph neural networks for detecting anomalies in scientific workflows},
  author={Jin, Hongwei and Raghavan, Krishnan and Papadimitriou, George and Wang, Cong and Mandal, Anirban and Kiran, Mariam and Deelman, Ewa and Balaprakash, Prasanna},
  journal={The International Journal of High Performance Computing Applications},
  volume={37},
  number={3-4},
  pages={394--411},
  year={2023},
  publisher={SAGE Publications Sage UK: London, England}
}
```
