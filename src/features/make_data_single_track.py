# @Author: Billy Li <billyli>
# @Date:   06-05-2022
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 06-05-2022



import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import uniform

data_dir = Path.cwd().parent.joinpath("data")
sys.path.insert(1, str(data_dir))

from hit_generator import stochastic
from util import plot_in_RAM, small_helix_check

def discretize(x, min, max, res):
    # return the discretized index of a value given a range and resolution
    step = (max-min)/res
    result = (x-min)//step
    if result >= res:
        result = res-1
    return int(result)

def xy2map(xs, ys, res):
    # return a z-t ptcl number map
    map = np.zeros(shape=(res,res), dtype=float)
    xmin, xmax = -810, 810
    ymin, ymax = -810, 810

    for x, y in zip(xs, ys):
        xIdx = discretize(x, xmin, xmax, res)
        yIdx = discretize(y, ymin, ymax, res)
        map[res-1-yIdx, xIdx] = 1.0

    return map

def make_data_single_track():

    ### set dataset property
    # Number of samples
    N_data = 100
    N_generated = 0
    # quality cut
    dx_min = 100
    dy_min = 100
    res = 256




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
        hit_dict = gen.generate(mode='production')
        if small_helix_check(hits,dx_min=dx_min,dy_min=dy_min):
            continue
        else:
            x = plot_in_RAM(hit_dict, res)
            x = x.reshape(res,res,4)
            plt.imshow(x)
            plt.show()
            N_generated += 1

    return

if __name__ == "__main__":
    make_data_single_track()
    return
