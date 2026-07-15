#!/usr/bin/env python3
"""
generate_default_map.py -- write maps/default_map.{yaml,pgm}

Reproduces the previous hardcoded arena (12m x 8m, 0.3m walls, 4 obstacle
rects) as a standard ROS map_server map, so the new static-map sim has a
default that behaves like the old one out of the box.

Run this once (`python3 generate_default_map.py`) or adapt the RECTS list /
dimensions to build your own map. Re-run any time you want to regenerate it.
"""
import os
import numpy as np

WIDTH, HEIGHT, RES = 12.0, 8.0, 0.05
WALL_T = 0.3
RECTS = [  # (x0, y0, x1, y1) in metres, same layout as the old sim
    (3.0, 3.0, 4.0, 5.2),
    (6.6, 1.0, 7.6, 3.2),
    (8.6, 4.6, 9.8, 6.6),
    (4.8, 5.6, 6.0, 6.8),
]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps')


def build_grid():
    nx, ny = int(round(WIDTH / RES)), int(round(HEIGHT / RES))
    occ = np.zeros((ny, nx), dtype=bool)

    def fill(x0, y0, x1, y1):
        i0, j0 = int(x0 / RES), int(y0 / RES)
        i1, j1 = int(x1 / RES), int(y1 / RES)
        occ[max(0, j0):min(ny, j1), max(0, i0):min(nx, i1)] = True

    fill(0, 0, WIDTH, WALL_T); fill(0, HEIGHT - WALL_T, WIDTH, HEIGHT)
    fill(0, 0, WALL_T, HEIGHT); fill(WIDTH - WALL_T, 0, WIDTH, HEIGHT)
    for r in RECTS:
        fill(*r)
    return occ


def write_pgm(path, occ):
    # map_server convention: 255 = free (white), 0 = occupied (black); file row 0
    # is the TOP of the image, i.e. the grid is flipped vertically on write.
    img = np.where(occ, 0, 255).astype(np.uint8)
    img = np.flipud(img)
    ny, nx = img.shape
    with open(path, 'wb') as f:
        f.write(f"P5\n{nx} {ny}\n255\n".encode('ascii'))
        f.write(img.tobytes())


def write_yaml(path, pgm_name):
    with open(path, 'w') as f:
        f.write(f"""image: {pgm_name}
resolution: {RES}
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
""")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    occ = build_grid()
    write_pgm(os.path.join(OUT_DIR, 'default_map.pgm'), occ)
    write_yaml(os.path.join(OUT_DIR, 'default_map.yaml'), 'default_map.pgm')
    print(f"Wrote {OUT_DIR}/default_map.yaml and default_map.pgm "
          f"({occ.shape[1]}x{occ.shape[0]} cells @ {RES} m/cell)")


if __name__ == '__main__':
    main()
