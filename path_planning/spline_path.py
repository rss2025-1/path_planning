import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import yaml
import os
import json
import numpy as np
def spline(path):
    x, y = zip(*path)
    x,y  = list(x),list(y) 
    tck, u = splprep([x, y], s=10)  # adjust s for smoothness
    x_smooth, y_smooth = splev(u, tck)
    return list(zip(x_smooth, y_smooth))
# Provided trajectory points
def example():

    num_points = 25
    x_start = 0
    x_end = 50
    noise_amplitude = 3

    # Generate increasing x values
    x = np.linspace(x_start, x_end, num_points)

    # Add jagged y values (random noise)
    np.random.seed(1)
    y = np.cumsum(np.random.uniform(-noise_amplitude, noise_amplitude, num_points))


    # Fit a parametric spline with smoothing
    tck, u = splprep([x, y], s=10)  # adjust s for smoothness

    # Evaluate the spline at more points for smooth curve
    u_fine = np.linspace(0, 1, 2*num_points)
    x_smooth, y_smooth = splev(u, tck)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro-', label='Original Points')
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Smoothed Spline')
    plt.legend()
    plt.title('Smoothed Trajectory Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    path = [
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
]
    smooth_path =spline(path)
    x_smooth, y_smooth = zip(*smooth_path)
    x, y = zip(*path)

    x,y  = list(x),list(y) 
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o--', label='Original Path')
    plt.plot(x_smooth, y_smooth, '-', linewidth=2, label='Smoothed Path')
    plt.axis('equal')
    plt.title("Smoothed 90° Turn Path")
    plt.legend()
    plt.grid(True)
    plt.show()