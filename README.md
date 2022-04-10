# Graph Neural Network for Anomaly Detection and Classification in Scientific Workflows


## Repo Content

data


### Step 0 : ETA
- visualize the graphs
- feature analysis

### Step 1: Preprocess the data


```python3
python3 preprocess_data.py --workflow_type all
```

Workflow type: nowcast-clustering-16
11070
Workflow type: 1000genome
72675
Workflow type: nowcast-clustering-8
15990
Workflow type: wind-clustering-casa
8610
Workflow type: wind-noclustering-casa
30914

Comment: raw data includes a number of invalid files from wind-nonclustering-casa

The script usage:
```
usage: 

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