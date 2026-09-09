"""Setup script for offline-photo-video-ai."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="offline-photo-video-ai",
    version="1.0.0",
    author="Photo-Video AI Team",
    description="Offline photo editing and image-to-video generation AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mfelix100/offline-photo-video-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "click>=8.0.0",
        "imageio>=2.33.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "photo-ai=cli:cli",
        ],
    },
)
