#!/usr/bin/env python3

import os
import pandas as pd

for folder in os.listdir("."):
    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):
        abs_path = "/".join([folder, filename])
        if not os.path.isfile(abs_path):
            continue

        if filename.startswith("1000-genome"):
            df = pd.read_csv(abs_path)
            anomaly_type = list(df[~df.anomaly_type.isin(["normal", "None"])]["anomaly_type"].unique())
            
            if len(anomaly_type) == 0 and folder != "normal":
                print("No anomalies found but folder isn't normal")
                continue
            elif len(anomaly_type) == 1 and folder != anomaly_type[0]:
                print(abs_path)
                print("Folder doesn't match the anomaly!")
                continue
            elif len(anomaly_type) > 1:
                print("More than one anomalies found!")
                continue
            
            if len(anomaly_type) == 0:
                print("{},{},{},{}".format("1000-genome", filename[filename.find("-run")+1:filename.find(".csv")], "normal", ""))
            else:
                anomaly_nodes = (list(df[df["anomaly_type"] != "normal"]["kickstart_hostname"].unique()))
                print("{},{},{},{}".format("1000-genome", filename[filename.find("-run")+1:filename.find(".csv")] ,anomaly_type[0], " ".join(anomaly_nodes)))
