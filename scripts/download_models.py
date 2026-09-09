"""Download pre-trained model weights."""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download and manage model weights."""
    
    MODELS = {
        "esrgan": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth",
            "size_mb": 68,
        },
        "denoiser": {
            "url": "https://github.com/cszn/DnCNN/releases/download/v1.0/dncnn_color_blind.pth",
            "size_mb": 25,
        },
        "rife": {
            "url": "https://github.com/hzwer/RIFE/releases/download/v4.0/flownet.pkl",
            "size_mb": 45,
        },
    }
    
    def __init__(self, model_dir: str = "./models/weights"):
        """Initialize downloader."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def download_all(self) -> None:
        """Download all models."""
        logger.info(f"Downloading models to: {self.model_dir}")
        
        total_size = sum(m["size_mb"] for m in self.MODELS.values())
        logger.info(f"Total download size: ~{total_size}MB")
        
        for model_name, model_info in self.MODELS.items():
            self.download_model(model_name)
    
    def download_model(self, model_name: str) -> None:
        """Download specific model."""
        if model_name not in self.MODELS:
            logger.error(f"Unknown model: {model_name}")
            return
        
        model_info = self.MODELS[model_name]
        output_path = self.model_dir / f"{model_name}.pth"
        
        if output_path.exists():
            logger.info(f"Model already exists: {output_path}")
            return
        
        logger.info(f"Downloading {model_name} (~{model_info['size_mb']}MB)...")
        logger.info(f"URL: {model_info['url']}")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_info["url"], output_path)
            logger.info(f"✓ Downloaded: {output_path}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            logger.info("Manual download instructions:")
            logger.info(f"1. Download from: {model_info['url']}")
            logger.info(f"2. Save to: {output_path}")
    
    def verify_models(self) -> dict:
        """Verify downloaded models."""
        logger.info("Verifying models...")
        
        status = {}
        for model_name in self.MODELS.keys():
            model_path = self.model_dir / f"{model_name}.pth"
            exists = model_path.exists()
            status[model_name] = exists
            
            if exists:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ {model_name}: OK ({size_mb:.1f}MB)")
            else:
                logger.warning(f"✗ {model_name}: MISSING")
        
        return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to download (esrgan, denoiser, rife)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/weights",
        help="Directory to save models"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing models"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(model_dir=args.model_dir)
    
    if args.verify:
        downloader.verify_models()
    elif args.model:
        downloader.download_model(args.model)
    else:
        downloader.download_all()
        downloader.verify_models()
