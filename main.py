import sys
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit
from PySide6.QtCore import Qt

class MatrixWidget(QWidget):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.cells = []
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        layout.setSpacing(6)

        for row in range(self.size):
            row_cells = []
            for col in range(self.size):
                cell = QLineEdit()
                cell.setAlignment(Qt.AlignCenter)
                cell.setFixedSize(50, 40)
                cell.setPlaceholderText("0")

                layout.addWidget(cell, row, col)
                row_cells.append(cell)

            self.cells.append(row_cells)

        self.setLayout(layout)

app = QApplication(sys.argv)
window = MatrixWidget(50)
window.resize(500, 500)
window.show()
sys.exit(app.exec())