"""
Main Photo Editor class for handling image processing operations.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PhotoEditor:
    """Main class for photo editing operations."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_dir: str = "./models/weights",
        half_precision: bool = False
    ):
        """
        Initialize PhotoEditor.
        
        Args:
            device: Device to use ("cuda" or "cpu")
            model_dir: Directory containing model weights
            half_precision: Use FP16 for reduced memory
        """
        self.device = device
        self.model_dir = Path(model_dir)
        self.half_precision = half_precision
        
        logger.info(f"PhotoEditor initialized on device: {device}")
        
        # Models will be loaded lazily
        self.upscaler = None
        self.denoiser = None
        self.style_transfer = None
        
    def upscale(
        self,
        input_path: str,
        output_path: str,
        scale: int = 4,
        tile_size: int = 400
    ) -> None:
        """
        Upscale image using ESRGAN.
        
        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image
            scale: Upscaling factor (2, 3, or 4)
            tile_size: Size of tiles for processing large images
        """
        try:
            logger.info(f"Upscaling image: {input_path}")
            
            # Load image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            logger.info(f"Image shape: {img.shape}")
            
            # Placeholder for actual ESRGAN upscaling
            # This would load and run the actual model
            upscaled = self._upscale_with_model(img, scale)
            
            # Save result
            cv2.imwrite(output_path, upscaled)
            logger.info(f"Upscaled image saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during upscaling: {str(e)}")
            raise
    
    def denoise(
        self,
        input_path: str,
        output_path: str,
        strength: float = 0.5
    ) -> None:
        """
        Denoise image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save denoised image
            strength: Denoising strength (0.0 to 1.0)
        """
        try:
            logger.info(f"Denoising image: {input_path}")
            
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            # Simple bilateral filter as placeholder
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Blend with original based on strength
            result = cv2.addWeighted(img, 1 - strength, denoised, strength, 0)
            
            cv2.imwrite(output_path, result)
            logger.info(f"Denoised image saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during denoising: {str(e)}")
            raise
    
    def adjust_colors(
        self,
        input_path: str,
        output_path: str,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue_shift: float = 0.0
    ) -> None:
        """
        Adjust image colors.
        
        Args:
            input_path: Path to input image
            output_path: Path to save adjusted image
            brightness: Brightness multiplier (0.0 to 2.0)
            contrast: Contrast multiplier (0.0 to 2.0)
            saturation: Saturation multiplier (0.0 to 2.0)
            hue_shift: Hue shift in degrees (-180 to 180)
        """
        try:
            logger.info(f"Adjusting colors: {input_path}")
            
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            # Convert BGR to HSV for saturation and hue adjustment
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Adjust saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            
            # Adjust hue
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Adjust brightness and value
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
            
            # Convert back to BGR
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Adjust contrast
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
            
            cv2.imwrite(output_path, result)
            logger.info(f"Color-adjusted image saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during color adjustment: {str(e)}")
            raise
    
    def apply_filter(
        self,
        input_path: str,
        output_path: str,
        filter_type: str = "blur"
    ) -> None:
        """
        Apply various filters to image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save filtered image
            filter_type: Type of filter (blur, sharpen, edge, sepia)
        """
        try:
            logger.info(f"Applying {filter_type} filter to: {input_path}")
            
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            if filter_type == "blur":
                result = cv2.GaussianBlur(img, (15, 15), 0)
            elif filter_type == "sharpen":
                kernel = np.array([[-1, -1, -1],
                                 [-1,  9, -1],
                                 [-1, -1, -1]]) / 1.0
                result = cv2.filter2D(img, -1, kernel)
            elif filter_type == "edge":
                result = cv2.Canny(img, 100, 200)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            elif filter_type == "sepia":
                kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
                result = cv2.transform(img, kernel)
                result = np.clip(result, 0, 255)
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            cv2.imwrite(output_path, result)
            logger.info(f"Filtered image saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during filtering: {str(e)}")
            raise
    
    def _upscale_with_model(self, img: np.ndarray, scale: int) -> np.ndarray:
        """
        Placeholder for actual ESRGAN upscaling.
        In production, this would load and run the actual model.
        """
        h, w = img.shape[:2]
        new_h, new_w = h * scale, w * scale
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        operation: str = "upscale",
        **kwargs
    ) -> None:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            operation: Operation to perform
            **kwargs: Additional arguments for the operation
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        logger.info(f"Processing {len(images)} images from {input_dir}")
        
        for img_file in images:
            try:
                output_file = output_path / img_file.name
                
                if operation == "upscale":
                    self.upscale(str(img_file), str(output_file), **kwargs)
                elif operation == "denoise":
                    self.denoise(str(img_file), str(output_file), **kwargs)
                elif operation == "adjust_colors":
                    self.adjust_colors(str(img_file), str(output_file), **kwargs)
                else:
                    logger.warning(f"Unknown operation: {operation}")
                    
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
                continue
