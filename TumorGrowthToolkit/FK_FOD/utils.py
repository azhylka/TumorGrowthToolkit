import itertools
import numpy as np

def get_direction_to_index():
    mapping = {(x, y, z): (idx if idx < 13 else idx-1) for idx, (x, y, z) in enumerate(itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)))
                        if not (x == 0 and y == 0 and z == 0)}
    return mapping

def get_index_to_direction():
    mapping = {(idx if idx < 13 else idx-1): (x, y, z) for idx, (x, y, z) in enumerate(itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)))
                        if not (x == 0 and y == 0 and z == 0)}
    return mapping

def extract_dominant_discrete_orientation(discrete_fod_distribution):
    masking = np.sum(discrete_fod_distribution, axis=-1) > 0
    main_directions = np.argmax(discrete_fod_distribution, axis=-1)
    max_amplitudes = np.max(discrete_fod_distribution, axis=-1)
    
    index2direction = get_index_to_direction()
    lookup_table = np.stack([index2direction[idx] for idx in sorted(index2direction.keys())]) # skip the center direction (0,0,0)
    
    dominant_orientations = np.abs(lookup_table[main_directions].astype(np.float64))
    dominant_orientations[np.logical_not(masking)] = 0
    dominant_orientations *= max_amplitudes[..., np.newaxis]
    dominant_orientations /= np.max(dominant_orientations)
    return dominant_orientations
