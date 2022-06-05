import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import uniform

vis_dir = Path.cwd().parent.joinpath('visualization')
sys.path.insert(1, str(vis_dir))

from hit_generator import stochastic
from plot import plot_in_RAM
from util import small_helix_check

def make_data_single_track():

    ### set dataset property
    # Number of samples
    N_data = 100
    N_generated = 0
    # quality cut
    dx_min = 100
    dy_min = 100
    res = 128




    # set track source (db files)
    track_dir = Path.cwd().parent.joinpath('data').joinpath('raw')
    db_list = ["train_CeEndpoint-mix-fromCSV_1.db",
               "train_CeEndpoint-mix-fromCSV_2.db",\
               "train_CeEndpoint-mix-fromCSV_3.db"]
    file_list = [track_dir.joinpath(db) for db in db_list]

    # set track distribution
    dist = uniform(loc=1, scale=0)

    # set track generator
    gen = stochastic(dist=dist, db_files=file_list, hitNumCut=20)


    while N_generated < N_data:
        hits = gen.generate(mode='production')
        if small_helix_check(hits,dx_min=dx_min,dy_min=dy_min):
            continue
        else:
            x = plot_in_RAM(hits, res)
            x = x.reshape(res,res,4)
            plt.imshow(x)
            plt.show()
            N_generated += 1

    return

if __name__ == "__main__":
    make_data_single_track()
    return
