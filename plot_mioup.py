# for plotting output pickle of manage_weights.py

import pickle
import matplotlib.pyplot as plt

run_num = '0514'

with open('{}_miou.p'.format(run_num), 'rb') as f:
    ep_2_miou = pickle.load(f)

maxy = -1
maxysx = -1
xs = []
ys = []
for k,v in ep_2_miou.items():
    xs.append(float(k))
    ys.append(float(v))
    if v > maxy:
        maxy = v
        maxysx = k

print('max of {} occurs at epoch {}'.format(maxy, maxysx))
plt.scatter(xs, ys)
plt.show()
