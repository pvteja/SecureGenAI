"""SecureGenAI: vulnerability-aware code generation and repair (reference implementation)."""

from .config import Config
from .model import SecureGenAI, build_model, load_model, save_model

__all__ = ["Config", "SecureGenAI", "build_model", "load_model", "save_model"]
__version__ = "0.1.0"
