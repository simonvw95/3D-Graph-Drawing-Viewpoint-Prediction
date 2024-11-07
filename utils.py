import numpy as np
import math


def rectangular_to_spherical(points):

    # convert rectangular coordinates (x,y,z) to spherical coordinates (Elevation, Azimuth, Distance)
    spherical = np.zeros_like(points)
    # elevation
    spherical[:, 0] = np.degrees(np.arctan2(points[:, 2], np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))))
    # azimuth angle
    spherical[:, 1] = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    spherical[:, 2] = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))

    return spherical


def spherical_to_rectangular(points):

    # convert spherical coordinates  (Elevation (theta), Azimuth (phi), Distance) to rectangular coordinates (x,y,z)

    elevation = np.radians(points[:, 0])
    azimuth = np.radians(points[:, 1])
    distance = points[:, 2]

    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)

    return np.array([x, y, z]).T


def fibonacci_sphere(samples = 1000):

    """
    Function to generate an arbitrary number of points, more or less equally distributed on the sphere surface
    """

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    points = np.array(points)

    return points
