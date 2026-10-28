"""aran: a small, CPU-only multi-cell link-adaptation sandbox (toy SINR -> MCS -> goodput model).

Nothing here reproduces, or should be compared with, any operator or vendor trial result."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, always
__version__ = "0.1.0"
