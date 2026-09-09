"""Basic usage examples for photo editing and video generation."""

from photo_ai.editor import PhotoEditor
from photo_ai.video_gen import VideoGenerator
import logging

logging.basicConfig(level=logging.INFO)


def example_upscale():
    """Example: Upscale an image."""
    print("\n=== Image Upscaling Example ===")
    
    editor = PhotoEditor(device="cuda")
    editor.upscale(
        input_path="input.jpg",
        output_path="upscaled.jpg",
        scale=4
    )
    print("✓ Upscaling complete!")


def example_denoise():
    """Example: Denoise an image."""
    print("\n=== Image Denoising Example ===")
    
    editor = PhotoEditor(device="cuda")
    editor.denoise(
        input_path="noisy.jpg",
        output_path="clean.jpg",
        strength=0.7
    )
    print("✓ Denoising complete!")


def example_color_adjustment():
    """Example: Adjust image colors."""
    print("\n=== Color Adjustment Example ===")
    
    editor = PhotoEditor(device="cuda")
    editor.adjust_colors(
        input_path="input.jpg",
        output_path="adjusted.jpg",
        brightness=1.1,
        contrast=1.2,
        saturation=0.9,
        hue_shift=15
    )
    print("✓ Color adjustment complete!")


def example_apply_filter():
    """Example: Apply filter to image."""
    print("\n=== Filter Application Example ===")
    
    editor = PhotoEditor(device="cuda")
    
    # Apply blur
    editor.apply_filter(
        input_path="input.jpg",
        output_path="blurred.jpg",
        filter_type="blur"
    )
    
    # Apply sharpen
    editor.apply_filter(
        input_path="input.jpg",
        output_path="sharpened.jpg",
        filter_type="sharpen"
    )
    
    print("✓ Filter application complete!")


def example_image_to_video():
    """Example: Convert image to video."""
    print("\n=== Image-to-Video Example ===")
    
    video_gen = VideoGenerator(device="cuda")
    video_gen.image_to_video(
        input_path="input.jpg",
        output_path="video.mp4",
        duration=5.0,
        fps=30,
        motion_type="pan"
    )
    print("✓ Video generation complete!")


def example_images_to_video():
    """Example: Create video from multiple images."""
    print("\n=== Multi-Image Video Example ===")
    
    video_gen = VideoGenerator(device="cuda")
    video_gen.images_to_video(
        image_dir="./images",
        output_path="slideshow.mp4",
        fps=30,
        transition="fade",
        duration_per_image=2.0
    )
    print("✓ Slideshow creation complete!")


def example_batch_processing():
    """Example: Batch process multiple images."""
    print("\n=== Batch Processing Example ===")
    
    editor = PhotoEditor(device="cuda")
    editor.batch_process(
        input_dir="./input_images",
        output_dir="./output_images",
        operation="upscale",
        scale=4
    )
    print("✓ Batch processing complete!")


if __name__ == "__main__":
    print("Offline Photo-Video AI - Usage Examples")
    print("========================================")
    
    # Uncomment to run examples
    # example_upscale()
    # example_denoise()
    # example_color_adjustment()
    # example_apply_filter()
    # example_image_to_video()
    # example_images_to_video()
    # example_batch_processing()
    
    print("\nExamples ready to use! Uncomment in the code to run.")
