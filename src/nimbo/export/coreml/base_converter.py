# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0
# Based on Anemll (https://github.com/Anemll/Anemll) - MIT License

"""
Abstract base class for Apple Neural Engine model converters.
"""

from abc import ABC, abstractmethod


class BaseConverter(ABC):
    """Abstract base class for Apple Neural Engine model converters.

    This class defines the interface for converting PyTorch models to CoreML
    format optimized for Apple Neural Engine execution.

    Subclasses must implement the `convert` method to provide model-specific
    conversion logic.

    Attributes:
        model: The PyTorch model to convert
    """

    def __init__(self, model):
        """Initialize the converter with a PyTorch model.

        Args:
            model: The PyTorch model to convert to CoreML format
        """
        self.model = model

    def preprocess(self):
        """Common preprocessing steps before conversion.

        This method prepares the model for conversion by:
        - Moving to the correct device
        - Setting to evaluation mode
        - Freezing parameters

        Can be overridden by subclasses for model-specific preprocessing.
        """
        pass

    @abstractmethod
    def convert(self):
        """Convert the model to CoreML format.

        Each model type must implement its own conversion logic.

        Returns:
            ct.models.MLModel: The converted CoreML model
        """
        pass

    def postprocess(self, num_workers=None):
        """Common postprocessing steps after conversion.

        This method applies optimizations like LUT quantization.

        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        pass
