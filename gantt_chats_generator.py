import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.utils import create_dir, process_args
import os
from tqdm import tqdm

# NOTE: `finished` is the new column
columns = ["ready", "wms_delay", "pre_script_delay",
           "queue_delay", "stage_in_delay", "runtime",
           "stage_out_delay", "post_script_delay", "finished"]

colors = ["#ea4335", "#fbbc03", "#33a853",
          "#740264", "#000000", "#4cd195",
          "#4285f4", "#ea8023", "#5f6368"]
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    args = process_args()

    # if not os.path.exists("gantt_chats"):
    #     os.mkdir()
    create_dir(os.path.join(CUR_DIR, "gantt_chats"))
    
    raw_files = glob.glob(f"./data/*/{args['workflow']}*.csv")
    
    for fn in tqdm(raw_files):
        folder = fn.split("/")[2]
        cur_folder = os.path.join(CUR_DIR, "gantt_chats", folder)
        create_dir(cur_folder)

        # processing original raw data
        df = pd.read_csv(fn, index_col=[0]).fillna(0)
        df['ready'] -= df['ready'].min()
        df['sum'] = df[['ready', 'wms_delay', 'queue_delay', 'runtime', 'post_script_delay']].sum(1)
        df['runtime'] = df['runtime'] - (df['stage_in_delay'] + df['stage_out_delay'])
        max_sum = df['sum'].max()
        # new column
        df['finished'] = max_sum - df['sum']

        values = df[columns].values
        lefts = np.insert(np.cumsum(values, axis=1), 0, 0, axis=1)[:, :-1]
        bottoms = np.arange(values.shape[0])

        fig, ax = plt.subplots(figsize=(50, 50), tight_layout=True)
        for idx, name in enumerate(columns):
            value = values[:, idx]
            left = lefts[:, idx]
            plt.bar(x=left, height=1.0, width=value, bottom=bottoms,
                    color=colors[idx], orientation="horizontal", label=name)
        plt.margins(0)
        plt.axis("off")

        plt.savefig(os.path.join(cur_folder, f"{''.join(fn.split('/')[-1].split('.')[:-1])}.png"))
        plt.close()
    