"""Project-wide path constants.

Source files live in src/; everything else (resources, checkpoints) is
addressed relative to the project root.
"""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
