"""PyQt5 GUI Application for Offline Photo-Video AI."""

import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QProgressBar, QTabWidget, QGroupBox,
    QFormLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QScrollArea
import cv2
import numpy as np

from photo_ai.editor import PhotoEditor
from photo_ai.video_gen import VideoGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """Worker thread for processing operations."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, operation, **kwargs):
        super().__init__()
        self.operation = operation
        self.kwargs = kwargs
    
    def run(self):
        """Run processing operation."""
        try:
            self.progress.emit(25)
            
            if self.operation == "upscale":
                editor = PhotoEditor(device="cuda")
                editor.upscale(**self.kwargs)
            elif self.operation == "denoise":
                editor = PhotoEditor(device="cuda")
                editor.denoise(**self.kwargs)
            elif self.operation == "adjust_colors":
                editor = PhotoEditor(device="cuda")
                editor.adjust_colors(**self.kwargs)
            elif self.operation == "image_to_video":
                video_gen = VideoGenerator(device="cuda")
                video_gen.image_to_video(**self.kwargs)
            
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class PhotoEditorTab(QWidget):
    """Tab for photo editing operations."""
    
    def __init__(self):
        super().__init__()
        self.input_image = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        
        # Input/Output section
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout()
        
        self.input_label = QLabel("No image selected")
        input_btn = QPushButton("Select Input Image")
        input_btn.clicked.connect(self.select_input_image)
        
        self.output_label = QLabel("output.jpg")
        output_btn = QPushButton("Select Output Location")
        output_btn.clicked.connect(self.select_output_location)
        
        file_layout.addRow("Input:", self.input_label)
        file_layout.addRow("", input_btn)
        file_layout.addRow("Output:", self.output_label)
        file_layout.addRow("", output_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Operation selection
        operation_group = QGroupBox("Select Operation")
        operation_layout = QFormLayout()
        
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(["Upscale", "Denoise", "Color Adjust", "Apply Filter"])
        self.operation_combo.currentTextChanged.connect(self.on_operation_changed)
        
        operation_layout.addRow("Operation:", self.operation_combo)
        operation_group.setLayout(operation_layout)
        layout.addWidget(operation_group)
        
        # Parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        # Upscale parameters
        self.scale_spinbox = QSpinBox()
        self.scale_spinbox.setRange(2, 4)
        self.scale_spinbox.setValue(4)
        params_layout.addRow("Scale Factor:", self.scale_spinbox)
        
        # Denoise parameters
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        params_layout.addRow("Denoising Strength:", self.strength_slider)
        
        # Color adjustment parameters
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        params_layout.addRow("Brightness:", self.brightness_slider)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        params_layout.addRow("Contrast:", self.contrast_slider)
        
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        params_layout.addRow("Saturation:", self.saturation_slider)
        
        # Filter selection
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Blur", "Sharpen", "Edge", "Sepia"])
        params_layout.addRow("Filter Type:", self.filter_combo)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Process button
        process_btn = QPushButton("Process Image")
        process_btn.clicked.connect(self.process_image)
        layout.addWidget(process_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def select_input_image(self):
        """Select input image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if path:
            self.input_image = path
            self.input_label.setText(Path(path).name)
    
    def select_output_location(self):
        """Select output location."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Image", "",
            "JPEG Image (*.jpg);;PNG Image (*.png);;All Files (*)"
        )
        if path:
            self.output_label.setText(path)
    
    def on_operation_changed(self, operation):
        """Handle operation change."""
        logger.info(f"Operation changed to: {operation}")
    
    def process_image(self):
        """Process image with selected operation."""
        if not self.input_image:
            logger.warning("No input image selected")
            return
        
        operation = self.operation_combo.currentText().lower()
        output = self.output_label.text()
        
        kwargs = {
            "input_path": self.input_image,
            "output_path": output
        }
        
        if operation == "upscale":
            kwargs["scale"] = self.scale_spinbox.value()
        elif operation == "denoise":
            kwargs["strength"] = self.strength_slider.value() / 100.0
        elif operation == "color adjust":
            kwargs["brightness"] = self.brightness_slider.value() / 100.0
            kwargs["contrast"] = self.contrast_slider.value() / 100.0
            kwargs["saturation"] = self.saturation_slider.value() / 100.0
        elif operation == "apply filter":
            kwargs["filter_type"] = self.filter_combo.currentText().lower()
        
        self.worker = ProcessingWorker(operation.replace(" ", "_"), **kwargs)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(lambda: logger.info("Processing complete!"))
        self.worker.error.connect(lambda e: logger.error(f"Error: {e}"))
        self.worker.start()


class VideoGeneratorTab(QWidget):
    """Tab for video generation."""
    
    def __init__(self):
        super().__init__()
        self.input_image = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout()
        
        self.input_label = QLabel("No image selected")
        input_btn = QPushButton("Select Input Image")
        input_btn.clicked.connect(self.select_input_image)
        
        self.output_label = QLabel("output.mp4")
        output_btn = QPushButton("Select Output Location")
        output_btn.clicked.connect(self.select_output_location)
        
        file_layout.addRow("Input Image:", self.input_label)
        file_layout.addRow("", input_btn)
        file_layout.addRow("Output Video:", self.output_label)
        file_layout.addRow("", output_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Video parameters
        params_group = QGroupBox("Video Parameters")
        params_layout = QFormLayout()
        
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(1.0, 60.0)
        self.duration_spinbox.setValue(5.0)
        self.duration_spinbox.setSuffix(" seconds")
        params_layout.addRow("Duration:", self.duration_spinbox)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSuffix(" fps")
        params_layout.addRow("Frame Rate:", self.fps_spinbox)
        
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["Pan", "Zoom", "Rotate", "None"])
        params_layout.addRow("Motion Type:", self.motion_combo)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Generate button
        generate_btn = QPushButton("Generate Video")
        generate_btn.clicked.connect(self.generate_video)
        layout.addWidget(generate_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def select_input_image(self):
        """Select input image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if path:
            self.input_image = path
            self.input_label.setText(Path(path).name)
    
    def select_output_location(self):
        """Select output location."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "",
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)"
        )
        if path:
            self.output_label.setText(path)
    
    def generate_video(self):
        """Generate video from image."""
        if not self.input_image:
            logger.warning("No input image selected")
            return
        
        kwargs = {
            "input_path": self.input_image,
            "output_path": self.output_label.text(),
            "duration": self.duration_spinbox.value(),
            "fps": self.fps_spinbox.value(),
            "motion_type": self.motion_combo.currentText().lower()
        }
        
        self.worker = ProcessingWorker("image_to_video", **kwargs)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(lambda: logger.info("Video generation complete!"))
        self.worker.error.connect(lambda e: logger.error(f"Error: {e}"))
        self.worker.start()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize main UI."""
        self.setWindowTitle("Offline Photo-Video AI")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(PhotoEditorTab(), "Photo Editor")
        tabs.addTab(VideoGeneratorTab(), "Video Generator")
        
        self.setCentralWidget(tabs)
        self.show()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
