"""
Command-line interface for photo editing and video generation.
"""

import click
import logging
from pathlib import Path
from photo_ai.editor import PhotoEditor
from photo_ai.video_gen import VideoGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Offline Photo Editing & Image-to-Video AI"""
    pass


@cli.command()
@click.option('--input', required=True, help='Input image path')
@click.option('--output', required=True, help='Output image path')
@click.option('--scale', default=4, type=int, help='Upscaling factor (2, 3, 4)')
@click.option('--tile-size', default=400, type=int, help='Tile size for processing')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def upscale(input, output, scale, tile_size, device):
    """Upscale image using ESRGAN."""
    try:
        editor = PhotoEditor(device=device)
        editor.upscale(input, output, scale=scale, tile_size=tile_size)
        click.echo(f"✓ Image upscaled successfully: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


@cli.command()
@click.option('--input', required=True, help='Input image path')
@click.option('--output', required=True, help='Output image path')
@click.option('--strength', default=0.5, type=float, help='Denoising strength (0.0-1.0)')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def denoise(input, output, strength, device):
    """Denoise image."""
    try:
        editor = PhotoEditor(device=device)
        editor.denoise(input, output, strength=strength)
        click.echo(f"✓ Image denoised successfully: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


@cli.command()
@click.option('--input', required=True, help='Input image path')
@click.option('--output', required=True, help='Output image path')
@click.option('--brightness', default=1.0, type=float, help='Brightness (0.0-2.0)')
@click.option('--contrast', default=1.0, type=float, help='Contrast (0.0-2.0)')
@click.option('--saturation', default=1.0, type=float, help='Saturation (0.0-2.0)')
@click.option('--hue-shift', default=0.0, type=float, help='Hue shift (-180 to 180)')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def adjust_colors(input, output, brightness, contrast, saturation, hue_shift, device):
    """Adjust image colors."""
    try:
        editor = PhotoEditor(device=device)
        editor.adjust_colors(
            input, output,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue_shift=hue_shift
        )
        click.echo(f"✓ Colors adjusted successfully: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


@cli.command()
@click.option('--input', required=True, help='Input image path')
@click.option('--output', required=True, help='Output image path')
@click.option('--filter', default='blur', help='Filter type (blur, sharpen, edge, sepia)')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def apply_filter(input, output, filter, device):
    """Apply filter to image."""
    try:
        editor = PhotoEditor(device=device)
        editor.apply_filter(input, output, filter_type=filter)
        click.echo(f"✓ Filter applied successfully: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


@cli.command()
@click.option('--input', required=True, help='Input image path')
@click.option('--output', required=True, help='Output video path')
@click.option('--duration', default=5.0, type=float, help='Video duration in seconds')
@click.option('--fps', default=30, type=int, help='Frames per second')
@click.option('--motion', default='pan', help='Motion type (pan, zoom, rotate, none)')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def image_to_video(input, output, duration, fps, motion, device):
    """Convert image to video."""
    try:
        video_gen = VideoGenerator(device=device)
        video_gen.image_to_video(input, output, duration=duration, fps=fps, motion_type=motion)
        click.echo(f"✓ Video generated successfully: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


@cli.command()
@click.option('--input-dir', required=True, help='Input directory with images')
@click.option('--output', required=True, help='Output video path')
@click.option('--fps', default=30, type=int, help='Frames per second')
@click.option('--transition', default='fade', help='Transition type (fade, slide, zoom)')
@click.option('--duration-per-image', default=2.0, type=float, help='Duration per image')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def images_to_video(input_dir, output, fps, transition, duration_per_image, device):
    """Create video from multiple images."""
    try:
        video_gen = VideoGenerator(device=device)
        video_gen.images_to_video(
            input_dir, output,
            fps=fps,
            transition=transition,
            duration_per_image=duration_per_image
        )
        click.echo(f"✓ Video created successfully: {output}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


@cli.command()
@click.option('--input-dir', required=True, help='Input directory')
@click.option('--output-dir', required=True, help='Output directory')
@click.option('--operation', default='upscale', help='Operation to perform')
@click.option('--scale', default=4, type=int, help='Scale factor for upscale')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
def batch(input_dir, output_dir, operation, scale, device):
    """Process multiple images."""
    try:
        editor = PhotoEditor(device=device)
        editor.batch_process(input_dir, output_dir, operation=operation, scale=scale)
        click.echo(f"✓ Batch processing completed: {output_dir}")
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)


if __name__ == '__main__':
    cli()
