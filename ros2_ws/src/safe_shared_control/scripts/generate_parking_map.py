#!/usr/bin/env python3
"""
generate_parking_map.py -- write maps/park_{tight,mid,wide}.{yaml,pgm}

Reverse-parking scenario: two trucks parked nose-in against a back wall with a
bay between them, and an open approach lane in front. The driver pulls along
the lane, then reverses into the bay -- the manoeuvre where an articulated
machine is hardest to control by hand, so it isolates what the CBF-MPC filter
is actually contributing.

Three gap widths are written so the same manoeuvre can be run at three
difficulty levels. What matters is not the gap itself but how much room the
vehicle CENTRELINE has, since every collision disc sits on it:

    centreline freedom = gap - 2 * (disc_radius + margin)

At the default disc_radius=1.5 / margin=0.1 that is gap - 3.2 m.

Run `python3 generate_parking_map.py`, or edit the constants below.
"""
import os

import numpy as np

WIDTH, HEIGHT, RES = 24.0, 20.0, 0.05
WALL_T = 0.4

TRUCK_W = 2.5           # truck width (across the bay)
TRUCK_L = 8.0           # truck length (into the bay)
BAY_X = 12.0            # bay centreline
BAY_Y0 = 7.5            # bay mouth
BAY_Y1 = BAY_Y0 + TRUCK_L
BACK_T = 0.4            # back wall thickness

VARIANTS = {            # name -> gap between the two trucks [m]
    'park_tight': 4.0,
    'park_mid': 4.5,
    'park_wide': 5.5,
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps')


def build_grid(gap):
    nx, ny = int(round(WIDTH / RES)), int(round(HEIGHT / RES))
    occ = np.zeros((ny, nx), dtype=bool)

    def fill(x0, y0, x1, y1):
        i0, j0 = int(x0 / RES), int(y0 / RES)
        i1, j1 = int(x1 / RES), int(y1 / RES)
        occ[max(0, j0):min(ny, j1), max(0, i0):min(nx, i1)] = True

    # arena walls
    fill(0, 0, WIDTH, WALL_T)
    fill(0, HEIGHT - WALL_T, WIDTH, HEIGHT)
    fill(0, 0, WALL_T, HEIGHT)
    fill(WIDTH - WALL_T, 0, WIDTH, HEIGHT)

    # back wall the trucks are parked against
    fill(WALL_T, BAY_Y1, WIDTH - WALL_T, BAY_Y1 + BACK_T)

    # the two trucks
    left_x1 = BAY_X - gap / 2.0
    right_x0 = BAY_X + gap / 2.0
    fill(left_x1 - TRUCK_W, BAY_Y0, left_x1, BAY_Y1)
    fill(right_x0, BAY_Y0, right_x0 + TRUCK_W, BAY_Y1)

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
    for name, gap in VARIANTS.items():
        occ = build_grid(gap)
        write_pgm(os.path.join(OUT_DIR, name + '.pgm'), occ)
        write_yaml(os.path.join(OUT_DIR, name + '.yaml'), name + '.pgm')
        print(f"Wrote {OUT_DIR}/{name}.yaml  (gap {gap:.1f} m, "
              f"centreline freedom {gap - 3.2:+.2f} m at r_disc=1.5, margin=0.1)")
    print(f"\n{occ.shape[1]}x{occ.shape[0]} cells @ {RES} m/cell "
          f"({occ.shape[0] * occ.shape[1] / 1e6:.2f} MB per OccupancyGrid)")


if __name__ == '__main__':
    main()
