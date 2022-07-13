import glob
import json
import os

import pandas as pd


def create_dir(path):
    """ Create a dir where the processed data will be stored

    Args:
        path (str): Path to create the folder.
    """
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


def process_data(graphs, drop_columns):
    """ Process the columns for graphs. """
    raise NotImplementedError


def parse_data(flag, json_path, classes):
    """ Parse the json file into graphs.

    Args:
        flag (str): Flag name.
        json_path (str): Json file path.
        classes (_type_): _description_

    Returns:
        dict: Graph with keys: y, edge_index, x
    """
    counter = 0
    edge_index = []
    lookup = {}
    graphs = []
    # columns = ['type', 'ready',
    #            'submit', 'execute_start', 'execute_end', 'post_script_start',
    #            'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
    #            'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']

    # REVIEW:
    # runtime = execute_end - execute_start
    # post_script_delay = post_script_end - post_script_start
    columns = ['type',
               'ready',
               'submit',
               #    'execute_start',
               #    'execute_end',
               #    'post_script_start',
               #    'post_script_end',
               'wms_delay',
               'pre_script_delay',
               'queue_delay',
               'runtime',
               'post_script_delay',
               'stage_in_delay',
               'stage_out_delay']

    # columns = ['type',
    #            'is_clustered',
    #            'runtime',
    #            'post_script_delay',
    #            'pre_script_delay',
    #            'queue_delay',
    #            'stage_in_delay',
    #            'stage_out_delay',
    #            'wms_delay',
    #            'stage_in_bytes',
    #            'stage_out_bytes',
    #            'kickstart_executables_cpu_time',
    #            'kickstart_status',
    #            'kickstart_executables_exitcode'
    #            ]
    # columns = ['type', 'ready', 'submit', 'wms_delay', 'pre_script_delay', 'queue_delay',
    #            'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']
    with open(json_path, "r") as f:
        adjacency_list = json.load(f)

        for l in adjacency_list:
            lookup[l] = counter
            counter += 1

        for l in adjacency_list:
            for e in adjacency_list[l]:
                edge_index.append([lookup[l], lookup[e]])
        for d in os.listdir("data"):
            for f in glob.glob(os.path.join("data", d, flag + "*")):
                try:
                    if d.split("_")[0] in classes:
                        graph = {"y": classes[d.split("_")[0]],
                                 "edge_index": edge_index,
                                 "x": []}
                        features = pd.read_csv(f, index_col=[0])
                        features = features.fillna(0)
                        # features = features.replace('', -1, regex=True)

                        for l in lookup:
                            if l.startswith("create_dir_") or l.startswith("cleanup_"):
                                new_l = l.split("-")[0]
                            else:
                                new_l = l
                            job_features = features[features.index.str.startswith(new_l)][columns].values.tolist()[0]

                            if len(features[features.index.str.startswith(new_l)]) < 1:
                                continue
                            if job_features[0] == 'auxiliary':
                                job_features[0] = 0
                            if job_features[0] == 'compute':
                                job_features[0] = 1
                            if job_features[0] == 'transfer':
                                job_features[0] = 2
                            # REVIEW: what's the line below
                            job_features = [-1 if x != x else x for x in job_features]
                            graph['x'].insert(lookup[l], job_features)

                        t_list = []
                        for i in range(len(graph['x'])):
                            t_list.append(graph['x'][i][1])
                        minim = min(t_list)

                        for i in range(len(graph['x'])):
                            lim = graph['x'][i][1:7]
                            lim = [v - minim for v in lim]
                            graph['x'][i][1:7] = lim
                            graphs.append(graph)
                except BaseException:
                    print("Error with the file's {} format.".format(f))
    return graphs
