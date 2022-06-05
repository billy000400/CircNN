import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

layout = {'pad':0, 'h_pad':0, 'w_pad':0, 'rect':(0,0,1,1) }

def small_helix_check(hit_dict, dx_min=5, dy_min=5):
    hits = [ it for k, it in hit_dict.items()]
    xs = []
    ys = []
    for hit in hits:
        xs.append(hit[0])
        ys.append(hit[1])

    xs = np.array(xs)
    ys = np.array(ys)
    dx = xs.max()-xs.min()
    dy = ys.max()-ys.min()

    return ((dx<dx_min)and(dy<dy_min))

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
    ax.scatter(xs, ys, c='b', s=1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()
    ax.set_xlim([-810, 810])
    ax.set_ylim([-810, 810])
    # Force a draw so we can grab the pixel buffer
    canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    x = np.array(canvas.renderer.buffer_rgba())
    x = np.rint(x[...,:3] @ [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    return np.array([x/255.0], dtype=np.float32)

def hits2arc(hit_dict):
    print("Arc Fitting")
    hits = [ it for k, it in hit_dict.items()]
    xs = []
    ys = []
    for hit in hits:
        xs.append(hit[0])
        ys.append(hit[1])
    xs = np.array(xs)
    ys = np.array(ys)
    n = xs.size

    a = 0
    b = 0
    R = 100

    # https://math.stackexchange.com/questions/2898295/how-to-quickly-fit-a-circle-by-given-random-arc-points
    print("assembling vec")
    vec = [np.sum(xs*xs*xs)+np.sum(xs*ys*ys),\
            np.sum(ys*ys*ys)+np.sum(xs*xs*ys),\
            np.sum(xs*xs)+np.sum(ys*ys)]
    vec = np.array(vec)
    print(vec.shape)
    print("assembling matrix")
    matrix = [[2*np.sum(xs*xs), 2*np.sum(xs*ys), np.sum(xs)],\
                [2*np.sum(xs*ys), 2*np.sum(ys*ys), np.sum(ys)],\
                [2*np.sum(xs), 2*np.sum(ys), n]]
    matrix = np.array(matrix)
    matrix_inv = np.linalg.inv(matrix)
    print(matrix_inv.shape)
    print("multiplication")
    [a, b, _] = matrix_inv.dot(vec)
    print("calculating c")
    c = (np.sum(xs*xs)+np.sum(ys*ys)-2*a*sum(xs)-2*b*sum(ys))/n
    print("calculating R")
    R = np.sqrt(c+a*a+b*b)
    return a, b, R
