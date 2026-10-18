"""NumPy batched rays with the upstream sensor's exact integer traversal rules."""
import numpy as np


def sensor_work_vectorized(robot_position, sensor_range, robot_belief, ground_truth):
    angles = []
    angle = 0.
    increment = .5 / 180 * np.pi
    while angle < 2 * np.pi:
        angles.append(angle)
        angle += increment
    angles = np.asarray(angles)
    x0, y0 = np.rint(robot_position).astype(np.int64)
    end_x = np.rint(robot_position[0] + np.cos(angles) * sensor_range).astype(np.int64)
    end_y = np.rint(robot_position[1] + np.sin(angles) * sensor_range).astype(np.int64)
    x = np.full(len(angles), x0, dtype=np.int64)
    y = np.full(len(angles), y0, dtype=np.int64)
    dx, dy = np.abs(end_x-x0), np.abs(end_y-y0)
    error = dx-dy
    dx, dy = dx*2, dy*2
    x_inc, y_inc = np.where(end_x>x0,1,-1), np.where(end_y>y0,1,-1)
    collisions = np.zeros(len(angles), dtype=np.int64)
    active = np.ones(len(angles), dtype=bool)
    while active.any():
        active &= (x>=0)&(x<ground_truth.shape[1])&(y>=0)&(y<ground_truth.shape[0])
        ids = np.flatnonzero(active)
        if not len(ids): break
        values = ground_truth[y[ids],x[ids]]
        walls = values==1
        collisions[ids] += walls
        stop = (walls&(collisions[ids]>=10)) | (~walls&(collisions[ids]>0)) | ((x[ids]==end_x[ids])&(y[ids]==end_y[ids]))
        active[ids[stop]] = False
        ids = ids[~stop]
        robot_belief[y[ids],x[ids]] = ground_truth[y[ids],x[ids]]
        horizontal = error[ids]>0
        a,b = ids[horizontal],ids[~horizontal]
        x[a] += x_inc[a]; error[a] -= dy[a]
        y[b] += y_inc[b]; error[b] += dx[b]
    return robot_belief
