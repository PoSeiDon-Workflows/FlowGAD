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



### Step 0 : ETA
- visualize the graphs
- feature analysis

### Step 1: Preprocess the data


```python3
python3 preprocess_data.py --workflow_type all
```



| Workflow Type          | Node # | 
|------------------------|:------:|
| nowcast-clustering-16  | 11070  | 
| 1000genome             | 72675  | 
| nowcast-clustering-8   | 15990  |  
| wind-clustering-casa   |  8610  |  
| wind-noclustering-casa | 30914  | 
| TOTAL                  |        |

Comment: raw data includes a number of invalid files from wind-nonclustering-casa_wind_wf experiment: 20200817T052029Z

The script usage:
```
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