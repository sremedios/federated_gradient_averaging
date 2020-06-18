import numpy as np
import random

def get_random_patch_indices(img_shape, patch_size):
    while True:
        st0 = random.randint(0, img_shape[0] - patch_size[0])
        en0 = st0 + patch_size[0]

        st1 = random.randint(0, img_shape[1] - patch_size[1])
        en1 = st1 + patch_size[1]

        st2 = random.randint(0, img_shape[2] - 1)
        en2 = st2 + 1

        yield slice(st0, en0), slice(st1, en1), slice(st2, en2)