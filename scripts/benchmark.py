"""Performance benchmarking script."""

import time
import logging
from pathlib import Path
from photo_ai.editor import PhotoEditor
from photo_ai.video_gen import VideoGenerator
from photo_ai.utils.device_utils import DeviceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Benchmark:
    """Benchmark performance of operations."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.results = {}
    
    def benchmark_upscale(self, input_image: str, scale: int = 4):
        """Benchmark image upscaling."""
        logger.info("Benchmarking upscale...")
        
        for device in ["cuda", "cpu"]:
            try:
                editor = PhotoEditor(device=device)
                
                start = time.time()
                editor.upscale(input_image, f"bench_upscale_{device}.jpg", scale=scale)
                elapsed = time.time() - start
                
                self.results[f"upscale_{device}"] = elapsed
                logger.info(f"  {device.upper()}: {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"  {device.upper()}: Failed - {str(e)}")
    
    def benchmark_denoise(self, input_image: str):
        """Benchmark image denoising."""
        logger.info("Benchmarking denoise...")
        
        for device in ["cuda", "cpu"]:
            try:
                editor = PhotoEditor(device=device)
                
                start = time.time()
                editor.denoise(input_image, f"bench_denoise_{device}.jpg")
                elapsed = time.time() - start
                
                self.results[f"denoise_{device}"] = elapsed
                logger.info(f"  {device.upper()}: {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"  {device.upper()}: Failed - {str(e)}")
    
    def benchmark_video_generation(self, input_image: str, duration: float = 5.0):
        """Benchmark video generation."""
        logger.info("Benchmarking video generation...")
        
        for device in ["cuda", "cpu"]:
            try:
                video_gen = VideoGenerator(device=device)
                
                start = time.time()
                video_gen.image_to_video(input_image, f"bench_video_{device}.mp4", duration=duration)
                elapsed = time.time() - start
                
                self.results[f"video_gen_{device}"] = elapsed
                logger.info(f"  {device.upper()}: {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"  {device.upper()}: Failed - {str(e)}")
    
    def print_summary(self):
        """Print benchmark summary."""
        logger.info("\n=== Benchmark Summary ===")
        logger.info(f"Device Info: {DeviceManager.get_device_info()}")
        logger.info("\nResults:")
        for operation, elapsed in self.results.items():
            logger.info(f"  {operation}: {elapsed:.2f}s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark photo editing operations")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--operation", type=str, default="all",
                       choices=["upscale", "denoise", "video", "all"],
                       help="Operation to benchmark")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        logger.error(f"Input image not found: {args.input}")
        exit(1)
    
    benchmark = Benchmark()
    
    if args.operation in ["upscale", "all"]:
        benchmark.benchmark_upscale(args.input)
    
    if args.operation in ["denoise", "all"]:
        benchmark.benchmark_denoise(args.input)
    
    if args.operation in ["video", "all"]:
        benchmark.benchmark_video_generation(args.input)
    
    benchmark.print_summary()
