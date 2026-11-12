"""Write ref_frame_nograd.npz: non-grad outputs of torchspin main 0.9.0 for the
configurations in torchspin/tests/test_frame_gradients.py::_nograd_outputs.

Run with the main 0.9.0 sources first on the path (the test module is loaded
by file path, so it need not exist in that tree):

    PYTHONPATH=/path/to/torchspin-0.9.0 OMP_NUM_THREADS=1 \\
        python3 tests/data/generate_frame_nograd_refs.py
"""
import importlib.util
import os
import platform

import numpy as np
import torch
import torchspin

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    '_frame_tests', os.path.join(here, '..', '..', 'torchspin', 'tests', 'test_frame_gradients.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out = mod._nograd_outputs()
out['_platform'] = np.array(platform.platform())
out['_torch'] = np.array(torch.__version__)
out['_torchspin'] = np.array(torchspin.__version__)
np.savez_compressed(os.path.join(here, 'ref_frame_nograd.npz'), **out)
print(f'{len(out) - 3} arrays from torchspin {torchspin.__version__} ({torchspin.__file__})')
