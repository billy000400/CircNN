import numpy as np

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
