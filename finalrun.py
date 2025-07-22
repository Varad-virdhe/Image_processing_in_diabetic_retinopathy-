import sys
import os
import cv2 # type: ignore
import numpy as np
from PyQt5.QtWidgets import ( # type: ignore
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class OCTLesionAutoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCT Lesion Area Calculator (mm² only)")

        self.image = None
        self.mask = None
        self.mm_per_px = 0.01172  # ✅ Hardcoded: 6.0 mm scan / 512 px = 0.01172 mm/px

        # ---------- UI ----------
        self.label_info = QLabel("Load an OCT image to highlight lesions.")
        self.btn_load = QPushButton("Load OCT Image")
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setEnabled(False)
        self.btn_save = QPushButton("Save Result")
        self.btn_save.setEnabled(False)

        self.fig = Figure()
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_mask = self.fig.add_subplot(122)
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.label_info)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_clear)
        button_layout.addWidget(self.btn_save)
        layout.addLayout(button_layout)

        # ---------- Connections ----------
        self.btn_load.clicked.connect(self.load_image)
        self.btn_clear.clicked.connect(self.reset_view)
        self.btn_save.clicked.connect(self.save_result)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select OCT Image", os.path.expanduser("~"),
            "Images (*.png *.jpg *.bmp *.tif)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.label_info.setText("❌ Failed to load image.")
            return

        self.image = img
        self.process_image()

    def process_image(self):
        img = self.image.copy()

        # Step 1: Bilateral filter
        blur = cv2.bilateralFilter(img, 9, 75, 75)

        # Step 2: Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 35, 5
        )

        # Step 3: Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Step 4: Contour filtering
        mask_clean = np.zeros_like(opened)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 3000:
                cv2.drawContours(mask_clean, [cnt], -1, 255, -1)

        # Step 5: Central retina ROI
        h, w = img.shape
        roi_mask = np.zeros_like(img)
        cv2.rectangle(roi_mask, (0, int(h * 0.2)), (w, int(h * 0.85)), 255, -1)
        final_mask = cv2.bitwise_and(mask_clean, roi_mask)

        self.mask = final_mask
        damaged_px = int(np.sum(final_mask > 0))

        # Step 6: Area in mm² only
        area_mm2 = damaged_px * (self.mm_per_px ** 2)
        area_txt = f"{area_mm2:.2f} mm²"

        # Step 7: Display results
        self.ax_orig.clear()
        self.ax_mask.clear()

        self.ax_orig.imshow(self.image, cmap='gray')
        self.ax_orig.set_title("Original")
        self.ax_orig.axis('off')

        self.ax_mask.imshow(self.image, cmap='gray')
        self.ax_mask.imshow(final_mask, cmap='Reds', alpha=0.5)
        self.ax_mask.set_title(f"Detected Lesions\nArea: {area_txt}")
        self.ax_mask.axis('off')

        self.canvas.draw()
        self.label_info.setText(f"✅ Damaged Area: {area_txt}")
        self.btn_clear.setEnabled(True)
        self.btn_save.setEnabled(True)

    def save_result(self):
        if self.image is None or self.mask is None:
            return

        # Convert grayscale to BGR
        img_color = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Create red overlay on lesion regions
        overlay = img_color.copy()
        overlay[self.mask > 0] = [0, 0, 255]

        blended = cv2.addWeighted(img_color, 0.7, overlay, 0.3, 0)

        # Ask where to save
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", os.path.expanduser("~"),
            "PNG Image (*.png)"
        )
        if path:
            cv2.imwrite(path, blended)
            self.label_info.setText("✅ Image saved successfully.")

    def reset_view(self):
        self.ax_orig.clear()
        self.ax_mask.clear()
        self.canvas.draw()

        self.image = None
        self.mask = None
        self.label_info.setText("Load an OCT image to highlight lesions.")
        self.btn_clear.setEnabled(False)
        self.btn_save.setEnabled(False)


# ---------- main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OCTLesionAutoApp()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())
