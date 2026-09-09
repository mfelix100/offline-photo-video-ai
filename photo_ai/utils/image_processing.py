"""Image processing utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            path: Image file path
            
        Returns:
            np.ndarray: Image array (BGR)
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return img
    
    @staticmethod
    def save_image(path: str, img: np.ndarray) -> None:
        """
        Save image to file.
        
        Args:
            path: Output file path
            img: Image array
        """
        cv2.imwrite(path, img)
        logger.info(f"Image saved: {path}")
    
    @staticmethod
    def resize_image(
        img: np.ndarray,
        size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image.
        
        Args:
            img: Input image
            size: Target size (width, height)
            maintain_aspect: Maintain aspect ratio
            
        Returns:
            np.ndarray: Resized image
        """
        if maintain_aspect:
            h, w = img.shape[:2]
            target_w, target_h = size
            aspect = w / h
            target_aspect = target_w / target_h
            
            if aspect > target_aspect:
                new_w = target_w
                new_h = int(target_w / aspect)
            else:
                new_h = target_h
                new_w = int(target_h * aspect)
            
            resized = cv2.resize(img, (new_w, new_h))
            
            # Pad to target size
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            return padded
        else:
            return cv2.resize(img, size)
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            img: Input image
            
        Returns:
            np.ndarray: Normalized image
        """
        return img.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(img: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255].
        
        Args:
            img: Normalized image
            
        Returns:
            np.ndarray: Denormalized image
        """
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
