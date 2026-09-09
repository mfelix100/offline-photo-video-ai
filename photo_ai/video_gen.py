"""
Video generation module for creating videos from images.
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple
import subprocess

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Class for generating videos from images."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        codec: str = "libx264",
        quality: str = "high"
    ):
        """
        Initialize VideoGenerator.
        
        Args:
            device: Device to use ("cuda" or "cpu")
            codec: Video codec (libx264, libx265)
            quality: Output quality (high, medium, low)
        """
        self.device = device
        self.codec = codec
        self.quality = quality
        
        # Quality presets
        self.quality_presets = {
            "high": {"bitrate": "5000k", "preset": "slow"},
            "medium": {"bitrate": "2500k", "preset": "medium"},
            "low": {"bitrate": "1000k", "preset": "fast"}
        }
        
        logger.info(f"VideoGenerator initialized with codec: {codec}, quality: {quality}")
    
    def image_to_video(
        self,
        input_path: str,
        output_path: str,
        duration: float = 5.0,
        fps: int = 30,
        motion_type: str = "pan",
        resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Convert a single image into a video with motion effects.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output video
            duration: Video duration in seconds
            fps: Frames per second
            motion_type: Type of motion (pan, zoom, rotate, none)
            resolution: Output resolution (width, height)
        """
        try:
            logger.info(f"Converting image to video: {input_path}")
            
            # Load image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            logger.info(f"Input image shape: {img.shape}")
            
            # Get or set resolution
            if resolution is None:
                h, w = img.shape[:2]
                # Ensure even dimensions for video codec
                w = w if w % 2 == 0 else w - 1
                h = h if h % 2 == 0 else h - 1
                resolution = (w, h)
            
            # Generate frames
            num_frames = int(duration * fps)
            frames = self._generate_motion_frames(img, num_frames, motion_type, resolution)
            
            # Write video
            self._write_video(frames, output_path, fps, resolution)
            
            logger.info(f"Video saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during image-to-video conversion: {str(e)}")
            raise
    
    def images_to_video(
        self,
        image_dir: str,
        output_path: str,
        fps: int = 30,
        transition: str = "fade",
        duration_per_image: float = 2.0
    ) -> None:
        """
        Create video from multiple images with transitions.
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save output video
            fps: Frames per second
            transition: Type of transition (fade, slide, zoom)
            duration_per_image: Duration each image is shown
        """
        try:
            logger.info(f"Creating video from images in: {image_dir}")
            
            image_path = Path(image_dir)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = sorted([f for f in image_path.iterdir() 
                           if f.suffix.lower() in image_extensions])
            
            if not images:
                raise ValueError(f"No images found in {image_dir}")
            
            logger.info(f"Found {len(images)} images")
            
            # Load all images
            frames_list = []
            resolution = None
            
            for img_file in images:
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.warning(f"Could not load image: {img_file}")
                    continue
                
                # Ensure uniform resolution
                if resolution is None:
                    h, w = img.shape[:2]
                    w = w if w % 2 == 0 else w - 1
                    h = h if h % 2 == 0 else h - 1
                    resolution = (w, h)
                else:
                    img = cv2.resize(img, resolution)
                
                # Generate transition frames
                transition_frames = self._generate_transition(
                    img, int(duration_per_image * fps), transition
                )
                frames_list.extend(transition_frames)
            
            # Write video
            self._write_video(frames_list, output_path, fps, resolution)
            
            logger.info(f"Video saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during multi-image video creation: {str(e)}")
            raise
    
    def _generate_motion_frames(
        self,
        img: np.ndarray,
        num_frames: int,
        motion_type: str,
        resolution: Tuple[int, int]
    ) -> list:
        """
        Generate frames with motion effects.
        
        Args:
            img: Input image
            num_frames: Number of frames to generate
            motion_type: Type of motion
            resolution: Output resolution
        
        Returns:
            List of frame arrays
        """
        frames = []
        h, w = img.shape[:2]
        target_w, target_h = resolution
        
        for i in range(num_frames):
            progress = i / num_frames
            
            if motion_type == "pan":
                # Pan from left to right
                offset_x = int((target_w - w) * progress)
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y1 = (target_h - h) // 2
                y2 = y1 + h
                x1 = offset_x
                x2 = x1 + w
                if x2 <= target_w:
                    frame[y1:y2, x1:x2] = img
                frames.append(frame)
                
            elif motion_type == "zoom":
                # Zoom in effect
                scale = 1.0 + progress * 0.5
                new_w = int(w * scale)
                new_h = int(h * scale)
                zoomed = cv2.resize(img, (new_w, new_h))
                
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y1 = (target_h - new_h) // 2
                y2 = y1 + new_h
                x1 = (target_w - new_w) // 2
                x2 = x1 + new_w
                
                if y2 <= target_h and x2 <= target_w:
                    frame[y1:y2, x1:x2] = zoomed
                else:
                    # Crop if needed
                    frame = zoomed[:target_h, :target_w]
                    
                frames.append(frame)
                
            elif motion_type == "rotate":
                # Rotation effect
                center = (w // 2, h // 2)
                angle = progress * 360
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h))
                
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y1 = (target_h - h) // 2
                y2 = y1 + h
                x1 = (target_w - w) // 2
                x2 = x1 + w
                frame[y1:y2, x1:x2] = rotated
                frames.append(frame)
                
            else:  # no motion
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y1 = (target_h - h) // 2
                y2 = y1 + h
                x1 = (target_w - w) // 2
                x2 = x1 + w
                frame[y1:y2, x1:x2] = img
                frames.append(frame)
        
        return frames
    
    def _generate_transition(
        self,
        img: np.ndarray,
        num_frames: int,
        transition_type: str
    ) -> list:
        """
        Generate transition frames for an image.
        
        Args:
            img: Input image
            num_frames: Number of frames
            transition_type: Type of transition
        
        Returns:
            List of frames
        """
        frames = []
        h, w = img.shape[:2]
        
        if transition_type == "fade":
            # Fade in effect
            for i in range(num_frames):
                alpha = i / num_frames
                frame = (img * alpha).astype(np.uint8)
                frames.append(frame)
                
        elif transition_type == "slide":
            # Slide in from left
            for i in range(num_frames):
                progress = i / num_frames
                frame = img.copy()
                frames.append(frame)
        else:
            # Default: static
            for _ in range(num_frames):
                frames.append(img.copy())
        
        return frames
    
    def _write_video(
        self,
        frames: list,
        output_path: str,
        fps: int,
        resolution: Tuple[int, int]
    ) -> None:
        """
        Write frames to video file using ffmpeg.
        
        Args:
            frames: List of frame arrays
            output_path: Output file path
            fps: Frames per second
            resolution: Video resolution
        """
        try:
            w, h = resolution
            
            # Get quality settings
            quality_settings = self.quality_presets.get(self.quality, self.quality_presets["high"])
            bitrate = quality_settings["bitrate"]
            preset = quality_settings["preset"]
            
            # Use ffmpeg for encoding
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{w}x{h}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', self.codec,
                '-preset', preset,
                '-b:v', bitrate,
                output_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write frames
            for frame in frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                process.stdin.write(frame.tobytes())
            
            process.stdin.close()
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read().decode()
                raise RuntimeError(f"FFmpeg error: {stderr}")
            
            logger.info(f"Video written successfully to {output_path}")
            
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install ffmpeg first.")
            raise RuntimeError("FFmpeg is required for video encoding")
