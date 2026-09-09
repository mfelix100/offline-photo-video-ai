# Offline Photo Editing & Image-to-Video AI

A powerful, fully offline photo editing and image-to-video generation tool powered by deep learning models. No internet connection required, no external API calls.

## Features

### Photo Editing
- **Image Upscaling**: 4x super-resolution using ESRGAN
- **Image Restoration**: Denoising and artifact removal
- **Style Transfer**: Apply artistic styles to photos
- **Color Adjustment**: Brightness, contrast, saturation, hue
- **Filters & Effects**: Blur, sharpen, edge detection, and more

### Image-to-Video
- **Motion Generation**: Create smooth videos from static images
- **Interpolation**: Frame interpolation for fluid motion
- **Animated Transitions**: Cinematic transitions between frames
- **Custom Duration**: Control video length and frame rate

### Performance
- **GPU Acceleration**: CUDA/cuDNN support for NVIDIA GPUs
- **CPU Fallback**: Works on CPU-only systems
- **Batch Processing**: Process multiple images efficiently
- **Low Memory Footprint**: Optimized model loading

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)
- 10GB disk space for models

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/mfelix100/offline-photo-video-ai.git
cd offline-photo-video-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download pre-trained models
```bash
python scripts/download_models.py
```

This will download all necessary model weights (~5-8GB).

## Quick Start

### Command Line Usage

#### Photo Upscaling
```bash
python cli.py upscale --input image.jpg --scale 4 --output upscaled.jpg
```

#### Image Denoising
```bash
python cli.py denoise --input noisy.jpg --output clean.jpg
```

#### Style Transfer
```bash
python cli.py style_transfer --input photo.jpg --style style.jpg --output result.jpg
```

#### Image-to-Video
```bash
python cli.py image_to_video --input image.jpg --duration 5 --fps 30 --output video.mp4
```

#### Color Adjustment
```bash
python cli.py adjust_colors --input image.jpg --brightness 1.2 --contrast 1.1 --saturation 0.9 --output adjusted.jpg
```

### Python API

```python
from photo_ai.editor import PhotoEditor
from photo_ai.video_gen import VideoGenerator

# Initialize editor
editor = PhotoEditor()

# Upscale an image
editor.upscale('input.jpg', output_path='upscaled.jpg', scale=4)

# Denoise
editor.denoise('noisy.jpg', output_path='clean.jpg')

# Generate video from image
video_gen = VideoGenerator()
video_gen.image_to_video('image.jpg', output_path='video.mp4', duration=5, fps=30)
```

### GUI Application

```bash
python gui/app.py
```

Launches an interactive desktop application with real-time preview.

## Project Structure

```
offline-photo-video-ai/
├── photo_ai/
│   ├── __init__.py
│   ├── editor.py              # Main photo editing class
│   ├── video_gen.py           # Video generation class
│   ├── models/
│   │   ├── esrgan.py          # Super-resolution model
│   │   ├── denoiser.py        # Denoising model
│   │   ├── style_transfer.py  # Style transfer model
│   │   ├── interpolation.py   # Frame interpolation
│   │   └── weights/           # Downloaded model weights
│   └── utils/
│       ├── image_processing.py
│       ├── video_processing.py
│       └── device_utils.py
├── gui/
│   ├── app.py                 # PyQt5 desktop application
│   ├── widgets/
│   └── assets/
├── cli.py                     # Command-line interface
├── scripts/
│   ├── download_models.py     # Model downloader
│   └── benchmark.py           # Performance benchmarks
├── examples/
│   ├── basic_usage.py
│   └── batch_processing.py
├── requirements.txt
├── setup.py
└── LICENSE
```

## Models Used

- **ESRGAN**: Real-ESRGAN for 4x super-resolution
- **Denoising**: DnCNN/FFDNet for image denoising
- **Style Transfer**: AdaIN or CycleGAN
- **Frame Interpolation**: RIFE or DAIN
- **Video Codec**: H.264/HEVC with libx264/libx265

## Performance

**On NVIDIA RTX 3080:**
- Upscaling 1080p image: ~2 seconds
- Image-to-video (5s @ 30fps): ~15 seconds
- Denoising: ~1 second

**On CPU (i7-9700K):**
- Upscaling 1080p image: ~45 seconds
- Image-to-video (5s @ 30fps): ~2 minutes
- Denoising: ~30 seconds

## Configuration

Edit `config.yaml` to customize:

```yaml
device: "cuda"  # or "cpu"
model_dir: "./models/weights"
batch_size: 4
video_codec: "libx264"
quality: "high"  # or "medium", "low"
```

## Advanced Features

### Batch Processing
```bash
python cli.py batch --input-dir ./images/ --operation upscale --scale 4 --output-dir ./results/
```

### Custom Model Loading
```python
from photo_ai.editor import PhotoEditor
editor = PhotoEditor(model_path='custom_models/')
```

### GPU Memory Optimization
```python
editor = PhotoEditor(half_precision=True)  # Use FP16 for reduced memory
```

## Limitations

- GPU memory requirements scale with image resolution
- Very high resolutions (>8K) may require tile-based processing
- Video generation quality depends on input image content
- Some effects may take longer on CPU

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## License

MIT License - See LICENSE file for details

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `config.yaml`
- Enable half-precision mode: `--half-precision`
- Use tile-based processing for large images

### Slow Performance
- Ensure CUDA is properly installed for GPU acceleration
- Check GPU memory availability with `nvidia-smi`
- Use lower quality settings for faster processing

### Model Download Issues
- Check internet connection and disk space
- Manually download models from releases
- Use `--model-dir` to specify custom path

## Disclaimer

This tool is designed for legitimate photo editing and creative purposes. Users are responsible for compliance with local laws and ethical use of generated content.

## Contact & Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Built with ❤️ for creative professionals and AI enthusiasts**
