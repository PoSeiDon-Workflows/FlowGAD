# Graph Neural Network for Anomaly Detection and Classification in Scientific Workflows

## setup env

* Install with CUDA available (default)
  
  `sh setup.sh gpu`
  
* Install with CPU only

  `sh setup.sh cpu`


## Data Statistics

graphs stat
|                        | # of nodes | # of edges | cpu samples | hdd samples | loss samples | normal samples |
| :--------------------- | :--------: | :--------: | :---------: | :---------: | :----------: | :------------: |
| 1000gemome             |     57     |    129     |     250     |     575     |     250      |      200       |
| nowcast-clustering-8   |     13     |     20     |     300     |     360     |     300      |      270       |
| nowcast-clustering-16  |     9      |     12     |     300     |     360     |     300      |      270       |
| wind-clustering-casa   |     7      |     8      |     300     |     360     |     300      |      270       |
| wind-noclustering-casa |     26     |     44     |     300     |     360     |     300      |      270       |

## Running examples

* graph-level anomaly classification
  `python example/demo_graph_classification.py`

* node-level anomaly classification
  `python example/demo_node_classification.py`

## Options

```bash
python examples/demo_graph_classification.py --help
usage: demo_graph_classification.py [-h] [--workflow {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,all}] [--binary] [--gpu GPU]
                                    [--epoch EPOCH] [--hidden_size HIDDEN_SIZE] [--batch_size BATCH_SIZE] [--train_size TRAIN_SIZE] [--lr LR] [--seed SEED] [--path PATH] [--logdir LOGDIR] [--force]
                                    [--verbose] [--output] [--anomaly_cat ANOMALY_CAT] [--anomaly_level [ANOMALY_LEVEL [ANOMALY_LEVEL ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --workflow {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,all}, -w {1000genome,nowcast-clustering-8,nowcast-clustering-16,wind-clustering-casa,wind-noclustering-casa,1000genome_new_2022,all}
                        Name of workflow.
  --binary              Toggle binary classification.
  --gpu GPU             GPU id. `-1` for CPU only.
  --epoch EPOCH         Number of epoch in training.
  --hidden_size HIDDEN_SIZE
                        Hidden channel size.
  --batch_size BATCH_SIZE
                        Batch size.
  --train_size TRAIN_SIZE
                        Train size [0.5, 1). And equal split on validation and testing.
  --lr LR               Learning rate.
  --seed SEED           Fix the random seed. `-1` for no random seed.
  --path PATH, -p PATH  Specify the root path of file.
  --logdir LOGDIR       Specify the log directory.
  --force               To force reprocess datasets.
  --verbose, -v         Toggle for verbose output.
  --output, -o          Toggle for pickle output file.
  --anomaly_cat ANOMALY_CAT
                        Specify the anomaly set.
  --anomaly_level [ANOMALY_LEVEL [ANOMALY_LEVEL ...]]
                        Specify the anomaly levels. Multiple inputs.
```