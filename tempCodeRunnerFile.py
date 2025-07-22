import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton,
    QVBoxLayout, QWidget, QLabel
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class LesionAreaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCT Lesion Area Calculator")
        self.setGeometry(100, 100, 800, 600)

        self.image = None
        self.drawing = False
        self.points = []

        self.pixel_spacing_mm = 0.006  # Example: 6 microns per pixel

        self.init_ui()

    def init_ui(self):
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()

        self.load_btn = QPushButton("Load OCT Image")
        self.load_btn.clicked.connect(self.load_image)

        self.clear_btn = QPushButton("Clear Lesion")
        self.clear_btn.clicked.connect(self.clear_points)

        self.area_label = QLabel("Draw a lesion to calculate area.")
        self.area_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.area_label)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.clear_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.show_image()

    def show_image(self):
        self.ax.clear()
        if self.image is not None:
            self.ax.imshow(self.image, cmap='gray')
        if self.points:
            pts = np.array(self.points)
            self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2)
        self.ax.axis('off')
        self.canvas.draw()

    def clear_points(self):
        self.points = []
        self.area_label.setText("Draw a lesion to calculate area.")
        self.show_image()

    def on_click(self, event):
        if event.inaxes != self.ax or self.image is None:
            return
        self.drawing = True
        self.points = [(event.xdata, event.ydata)]

    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax:
            self.points.append((event.xdata, event.ydata))
            self.show_image()

    def on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        self.calculate_area()

    def calculate_area(self):
        if len(self.points) < 3:
            self.area_label.setText("Draw a closed region.")
            return
        pts = np.array(self.points, dtype=np.int32)
        mask = np.zeros(self.image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        pixel_area = np.sum(mask)
        mm2_area = pixel_area * (self.pixel_spacing_mm ** 2)
        self.area_label.setText(f"Lesion Area: {mm2_area:.4f} mm²")
        self.show_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LesionAreaApp()
    win.show()
    sys.exit(app.exec_())
