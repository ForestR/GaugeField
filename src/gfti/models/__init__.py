"""Models for GFTI experiments."""

from .mlp import BaselineMLP, create_baseline
from .prototype import GFTIPrototype

__all__ = ["BaselineMLP", "create_baseline", "GFTIPrototype"]
