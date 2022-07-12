# miscellaneous functions

import numpy as np
from typing import Tuple
from itertools import accumulate

def get_unique_indices(
    a: np.ndarray
) -> Tuple[np.ndarray]:
    uniques, counts = np.unique(a, return_counts=True)
    return uniques, np.split(np.argsort(a), list(accumulate(counts))[:-1])