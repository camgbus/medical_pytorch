import numpy as np
import torch
from mp.utils.create_patches import patchify

def test_patchify():
    a = np.array([[[1, 4, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]],
                  [[1, 1, 1, 1, 2, 2],
                   [1, 6, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]],
                  [[1, 1, 1, 1, 2, 2],
                   [1, 2, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]],
                  [[1, 1, 1, 1, 2, 2],
                   [1, 2, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]],
                  [[1, 1, 1, 1, 2, 2],
                   [1, 2, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]],
                  [[1, 1, 1, 1, 2, 2],
                   [1, 2, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]],
                  [[1, 1, 1, 1, 2, 2],
                   [1, 2, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2],
                   [1, 1, 1, 1, 2, 2]]])
                   
    a = torch.from_numpy(a).unsqueeze(0).permute(1, 0, 2, 3)

    b = patchify(a, (1, 4, 4, 5), 0.5)
    c = patchify(a, (1, 3, 5, 3), 0.5)

    # Test 1
    assert len(b) == 2 and b[0].shape == (1, 4, 4, 5) and b[1].shape == (1, 4, 4, 5),\
                       "Number of patches or patch dimensions are not as expected."

    # Test 2
    assert len(c) == 4 and c[0].shape == (1, 3, 5, 3) and c[1].shape == (1, 3, 5, 3)\
                       and c[2].shape == (1, 3, 5, 3) and c[3].shape == (1, 3, 5, 3),\
                       "Number of patches or patch dimensions are not as expected."
    
    # Test 3
    try:
        patchify(a, (1, 3, 5, 3), 0.2)
        assert False, "Expected an AssertionError due to possible 0 division."
    except Exception as ex:
        if type(ex).__name__ != "AssertionError":
            assert False, "Expected an AssertionError due to possible 0 division."

    # Test 4
    try:
        patchify(a, (1, 3, 5, 3), 1)
        assert False, "Expected an AssertionError due to a desired overlap of 100%."
    except Exception as ex:
        if type(ex).__name__ != "AssertionError":
            assert False, "Expected an AssertionError because of 100% overlap."