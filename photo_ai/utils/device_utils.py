"""Device management utilities."""

import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and optimization."""
    
    @staticmethod
    def get_device(device_name: str = "cuda") -> torch.device:
        """
        Get appropriate device.
        
        Args:
            device_name: Requested device ("cuda" or "cpu")
            
        Returns:
            torch.device: Selected device
        """
        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")
    
    @staticmethod
    def get_device_info() -> dict:
        """
        Get device information.
        
        Returns:
            dict: Device information
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["total_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["allocated_memory"] = torch.cuda.memory_allocated(0) / 1e9
        
        return info
    
    @staticmethod
    def cleanup_memory():
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned")
