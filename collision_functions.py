"""A collection of functions used to simulate the particles."""

import numpy as np

from itertools import combinations
from matplotlib.collections import PathCollection
from numba import njit


def animate(frame: int,
            im: PathCollection,
            position: np.ndarray,
            velocity: np.ndarray,
            delta_time: float,
            maxes: tuple,
            radius: float
            ) -> tuple:
    """
    Creates the next image for the animation.
    
    Parameters
    ----------
    frame : int
        The frame number.
    im : PathCollection
        The image of the last frame.
    position : ndarray
        The array containing the xy coordinate pairs of each particle.
    velocity : ndarray
        The array containing the velocities of each particle.
    delta_time : float
        The time step per frame.
    maxes : tuple of float
        The bounds of the animation box.
    radius : float
        The radius of the particles.
    
    Returns
    -------
    im : PathCollection
        The next frame of the animation.
    """
    position = update(position, velocity, delta_time, maxes, radius)
    im.set_offsets(position)
    return im, 


@njit(cache=True)
def collision(indices: np.ndarray,
              distances: np.ndarray,
              position: np.ndarray,
              velocity: np.ndarray
              ) -> None:
    """
    Calculates the resultant velocities of a pair of colliding particles,
    modifying the velocity in-place.
    
    Parameters
    ----------
    indices : ndarray
        The array of indices of the colliding particles.
    distances : ndarray
        The distances between the colliding particles.
    position : ndarray
        The array containing the xy coordinate pairs of each colliding particle.
    velocity : ndarray
        The array containing the velocities of each colliding particle.
    """
    i = indices[:, 0]
    j = indices[:, 1]

    diff = position[i] - position[j]
    dot = np.sum(diff*(velocity[i] - velocity[j]), -1)[:, np.newaxis]

    velocity[i] -= dot / distances[:, np.newaxis]*diff
    velocity[j] += dot / distances[:, np.newaxis]*diff


def update(position: np.ndarray,
           velocity: np.ndarray,
           delta_time: float,
           maxes: tuple, 
           radius: float
           ) -> np.ndarray:
    """
    Updates the positions and velocities of the particles each frame.
    
    Parameters
    ----------
    position : ndarray
        The array containing the xy coordinate pairs of each particle.
    velocity : ndarray
        The array containing the velocities of each particle.
    delta_time : float
        The time step per frame.
    maxes : tuple of float
        The bounds of the animation box.
    radius : float
        The radius of the particles.
    
    Returns
    -------
    position : ndarray
        The array of updated positions for each particle.
    """
    position += delta_time*velocity

    # Flip velocities of out-of-bounds particles
    velocity[(position < 0) | ((position > maxes[0]) & (position > maxes[1]))] *= -1

    # Find which particles should collide
    distances = np.asarray(list(combinations(position, 2))).swapaxes(1, 2)
    distances = (np.diff(distances, axis=-1).squeeze() ** 2).sum(-1)
    indices = np.where(distances < 4*radius ** 2)[0]

    # Create combinations of colliding particles
    combination = np.asarray(list(combinations(np.arange(0, position.shape[0]), 2)))

    collision(combination[indices], distances[indices], position, velocity)

    return position
