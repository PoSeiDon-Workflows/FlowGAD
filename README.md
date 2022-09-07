# Graph Neural Network for Anomaly Detection and Classification in Scientific Workflows

## Repo Content

- **adjacency_list_dags**: json files with dependencies between nodes in each workflow
- **data**: raw data - characterizations of jobs in each workflow during multiple execustions
- **deepHyp_scripts**: scripts connected to finding best hyperparameters for GNN
- **helpers**: helper scripts that include functions and class definitions (models)
- **notebooks**: notebooks used during the code development
- **pickles**: pickle data from previous experiments (processsed raw data)
- **preprocess_graph_data**: pickles with workflow data in format expected by PyG
- **results**: csv files with results
- **submission_scripts**: bash scripts for jobs submissions

## setup env

* Install with CUDA available (default)
  
  `sh setup.sh gpu`
  
* Install with CPU only

  `sh setup.sh cpu`

### Step 0 : ETA

- visualize the graphs
- feature analysis

### Step 1: Preprocess the data

```python3
python3 preprocess_data.py --workflow_type all
```

graphs stat
|                        | cpu samples | hdd samples | loss samples | normal samples | # of nodes | # of edges |
| :--------------------- | :---------: | :---------: | :----------: | :------------: | ---------: | ---------: |
| 1000gemome             |     250     |     575     |     250      |      200       |         57 |        129 |
| nowcast-clustering-8   |     300     |     360     |     300      |      270       |         13 |         20 |
| nowcast-clustering-16  |     300     |     360     |     300      |      270       |          9 |         12 |
| wind-clustering-casa   |     300     |     360     |     300      |      270       |          7 |          8 |
| wind-noclustering-casa |     300     |     360     |     300      |      270       |         26 |         44 |

Comment: raw data includes a number of invalid files from wind-nonclustering-casa_wind_wf experiment: 20200817T052029Z

The script usage:

```text
usage: preprocess_data.py [-h] [--workflow_type WORKFLOW_TYPE]

Data preprocessing for GNN

options:
  -h, --help            show this help message and exit
  --workflow_type WORKFLOW_TYPE
                        name of a workflow: nowcast-clustering-16,1000genome,nowcast-
                        clustering-8,wind-clustering-casa,wind-noclustering-casa or ALL 

```

Output:

### Step 2: Train a model

```python3
python3 train_model.py --workflow_type all
```

The script usage:

```
usage: 

```

Output:



## Data description
| workflow   | # nodes per graph | # edges per graph |
| ---------- | ----------------- | ----------------- |
| 1000genome | 57                | 129               |