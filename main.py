import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtCore import Qt, QTimer

class WaveWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    def tick(self):
        self.phase += 0.2
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.blue, 2)
        painter.setPen(pen)

        w, h = self.width(), self.height()
        mid = h // 2

        prev = None
        for x in range(w):
            y = mid + 50 * np.sin(0.03 * x - self.phase)
            if prev:
                painter.drawLine(int(prev[0]), int(prev[1]), x, int(y))
            prev = (x, y)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaveWidget()
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())
