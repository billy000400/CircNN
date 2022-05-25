import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

layout = {'pad':0, 'h_pad':0, 'w_pad':0, 'rect':(0,0,1,1) }

def plot_in_RAM(hit_dict, resolution):
    hits = [ it for k, it in hit_dict.items()]
    xs = []
    ys = []
    for hit in hits:
        xs.append(hit[0])
        ys.append(hit[1])

    fig = Figure(figsize=(8,8), dpi=resolution/8, frameon=False, tight_layout=layout)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.scatter(xs, ys, c='k', s=1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()
    ax.set_xlim([-810, 810])
    ax.set_ylim([-810, 810])
    # Force a draw so we can grab the pixel buffer
    canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    x = np.array(canvas.renderer.buffer_rgba())
    return np.array([x/255.0], dtype=np.float32)
