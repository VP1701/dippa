from afs_model import AFSModel
import numpy as np

m = AFSModel(1.1059, 0.9858, 0.28, [(0.5, 0.0)], 0.2)
q = np.array([1.0, 2.0, 0.7, 0.4])      # nonzero theta AND gamma
u = np.array([0.8, 0.3])                # nonzero v AND omega
A, B, _ = m.linearize(q, u)
for i in range(4):
    d = np.zeros(4); d[i] = 1e-6
    print('q', i, np.abs((m.f(q+d,u) - m.f(q-d,u))/2e-6 - A[:,i]).max())
for j in range(2):
    d = np.zeros(2); d[j] = 1e-6
    print('u', j, np.abs((m.f(q,u+d) - m.f(q,u-d))/2e-6 - B[:,j]).max())