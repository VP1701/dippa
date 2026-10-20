
import math
import numpy as np 


def interpolate_path(goal, start, ds, theta=0.0, type="line"):
    x_g, y_g = goal[0], goal[1]
    x_f, y_f = start[0], start[1]
    dx = x_g-x_f
    dy = y_g-y_f
    
    
    if type == "line":
        n = int(np.ceil(np.hypot(dx, dy)/ds))+1
        t = np.linspace(0.0, 1.0, n)
        x = x_f + t * (dx)
        y = y_f + t * (dy)
        path = np.column_stack(x,y)
    else:

