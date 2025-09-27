"""
Dataset Module

Provides MRAG-Bench dataset loading, preprocessing, and perspective change scenario filtering.
Implements memory-efficient data streaming and batch processing.
"""

from .interface import DatasetInterface

__all__ = ["DatasetInterface"]