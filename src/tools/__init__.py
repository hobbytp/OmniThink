# This __init__.py is intended to expose components from subdirectories
# for easier access, e.g., src.tools.MindMap instead of src.tools.dspy.mindmap.MindMap

# Assuming the dspy versions are the defaults to be exposed directly under src.tools
# if specific framework versions are needed, they can be imported via their full path.

from .dspy.lm import *
from .dspy.mindmap import * # This should make MindMap class available if defined in dspy.mindmap
from .dspy.rm import *

# To specifically control what's exported and avoid wildcard imports:
# from .dspy.mindmap import MindMap # Example if MindMap is the only thing needed from there
# from .dspy.lm import SomeLM # Example
