"""
humanoid-walk

Observation Space
The observation space is a 67-dimensional vector containing information about:
- Center of mass velocity
- Positions of extremities (hands and feet)
- Head height
- Joint angles for all 21 joints
- Torso orientation
- Velocities of various body parts

Action Space
The action space is a 21-dimensional vector, with each dimension controlling a different joint in the humanoid body. Each action value ranges from -1.0 to 1.0.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def make_env():
    # Create environment with state observations
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env
