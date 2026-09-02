#######################################################################
# Name: ss_realistic_model.py
# Realistic signal strength communication model.
# References: https://hal.science/hal-03365129/document
#######################################################################

import numpy as np
import random
from skimage.draw import line

class SS_realistic_model:

    def __init__(self, P_T=-20, threshold_ss=-70, gamma=2, gamma_obst=4, dist_o=35, PL_o=31, X_g_min=0, X_g_max=0, K_min=0, K_max=0):
        self.P_T = P_T
        self.threshold_ss = threshold_ss
        self.gamma = gamma
        self.gamma_obst = gamma_obst
        self.dist_o = dist_o
        self.PL_o = PL_o

        # Randomize noise if min != max
        self.X_g = random.uniform(X_g_min, X_g_max) if X_g_min != X_g_max else X_g_min
        self.K = random.uniform(K_min, K_max) if K_min != K_max else K_min

    def get_signal_metrics(self, robot_belief, robot_a_location, robot_b_location):
        """Return continuous signal metrics for the link between two locations."""
        X, Y = line(robot_a_location[0], robot_a_location[1], robot_b_location[0], robot_b_location[1])

        # Count the number of obstacles and free in the line
        num_obst = 0
        wall_crossings = 0
        in_obstacle = False
        for (x, y) in zip(X[1:], Y[1:]):    # ignore first pose (source)
            if robot_belief[y, x] != 255:   # Obstacle & Unknown
                num_obst += 1
                if not in_obstacle:
                    wall_crossings += 1
                    in_obstacle = True
            else:
                in_obstacle = False
        total_dist = np.linalg.norm(robot_a_location - robot_b_location)
        dist_obst = num_obst * 1               # * self.map_resolution (Assume res = 1m/px)
        dist_free = max(total_dist - dist_obst, 0)

        # Compute path loss
        obstacle_loss = 0
        if dist_obst > 0:
            obstacle_loss = 10 * self.gamma_obst * np.log10(dist_obst) + self.K

        free_space_loss = 0
        if dist_free >= self.dist_o:
            free_space_loss = 10 * self.gamma * np.log10(dist_free/self.dist_o) + self.X_g

        # Compute received SS
        path_loss = self.PL_o + obstacle_loss + free_space_loss
        rssi = self.P_T - path_loss

        return {
            'rssi': float(rssi),
            'margin': float(rssi - self.threshold_ss),
            'path_loss': float(path_loss),
            'obstacle_loss': float(obstacle_loss),
            'free_space_loss': float(free_space_loss),
            'distance': float(total_dist),
            'obstacle_distance': float(dist_obst),
            'free_distance': float(dist_free),
            'wall_crossings': wall_crossings,
        }

    def compute_rssi(self, robot_belief, robot_a_location, robot_b_location):
        """Return received signal strength in dBm."""
        return self.get_signal_metrics(robot_belief, robot_a_location, robot_b_location)['rssi']

    def is_within_signal_strength(self, robot_belief, robot_a_location, robot_b_location):
        """Check whether the continuous RSSI is above the connection threshold."""
        return self.compute_rssi(robot_belief, robot_a_location, robot_b_location) > self.threshold_ss

