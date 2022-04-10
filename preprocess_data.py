import json
import glob
import os
import pandas as pd
import numpy as np
import pickle
import random
import argparse

from helpers.parsing_functions import parse_data

# for debbugging 
from IPython import embed


# for some level of reproducibility
random.seed(12345)
np.random.seed(12345)


def get_arguments():
    
    parser = argparse.ArgumentParser(description="Data preprocessing for GNN")    
    parser.add_argument('-workflow_type', type=str, default="ALL", help='name of a workflow: nowcast-clustering-16,'
        + '1000genome,nowcast-clustering-8,wind-clustering-casa,wind-noclustering-casa or ALL')   
    args = parser.parse_args()
    
    return args



def create_dir(path):
    '''Create a dir where the processed data will be stored'''
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


def load_data(flag):

    classes = {"normal": 0}
    counter = 1
    json_path = ""
    
    for d in os.listdir("data"):
        d = d.split("_")[0]       
        if d in classes:
            continue
        classes[d] = counter        
        counter += 1
        
    if flag == "nowcast-clustering-16":
        json_path = "adjacency_list_dags/casa_nowcast_clustering_16.json"
    elif flag == "1000genome":
        json_path = "adjacency_list_dags/1000genome.json"
    elif flag =="nowcast-clustering-8":
        json_path = "adjacency_list_dags/casa_nowcast_clustering_8.json"
    elif flag == "wind-clustering-casa":
        json_path = "adjacency_list_dags/casa_wind_clustering.json"
    elif flag == "wind-noclustering-casa":
        json_path = "adjacency_list_dags/casa_wind_no_clustering.json"
        
    graphs = parse_data(flag, json_path, classes)
    return graphs


def preprocess_data(wf_name, path, workflows_names):

    print("Workflow type: {}".format(wf_name))   
    # parse data from raw files
    graphs = []

    if wf_name == "ALL":
        for wf in workflows_names:
            graphs.append(load_data(wf))
    else:
        graphs = load_data(wf_name)
    print(len(graphs))
    with open(path +'graph_all_'+ str(wf_name) + '.pkl','wb') as f:
        pickle.dump(graphs, f)


def main():
    args = get_arguments()
    wf_name = args.workflow_type

    workflows_names = [ "nowcast-clustering-16","1000genome", "nowcast-clustering-8",
              "wind-clustering-casa","wind-noclustering-casa"]
    if wf_name not in workflows_names and wf_name != "ALL":
        print("Invalid workflow name.")
        exit()

    path = "preprocess_graph_data/"
    create_dir(path)
    preprocess_data(wf_name, path, workflows_names)

    return

if __name__ == '__main__':
	main()


