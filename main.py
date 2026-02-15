import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QFont, QIntValidator,
    QPainter, QPen
)

# ================= CONFIG =================
CELL_W = 50
CELL_H = 36
CELL_SPACING = 6

BUTTON_W = 220
BUTTON_H = 70

FONT_CELL = QFont("Consolas", 13)
FONT_OP = QFont("Consolas", 20)


# ================= MATRIX CELL =================
class MatrixCell(QLineEdit):
    def __init__(self, readonly=False, is_last=False, on_last_enter=None):
        super().__init__("0")

        self.is_last = is_last
        self.on_last_enter = on_last_enter

        self.setFont(FONT_CELL)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(CELL_W, CELL_H)
        self.setReadOnly(readonly)

        if not readonly:
            validator = QIntValidator(-99, 99, self)
            self.setValidator(validator)
            self.setMaxLength(3)  # -99 là 3 ký tự
            self.setAcceptDrops(False)

    def focusInEvent(self, event):
        if not self.isReadOnly() and self.text() == "0":
            self.clear()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if not self.isReadOnly():
            if not self.hasAcceptableInput():
                self.setText("0")
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.is_last and self.on_last_enter:
                self.on_last_enter()
            else:
                self.focusNextChild()
        else:
            super().keyPressEvent(event)


# ================= BRACKET =================
class BracketWidget(QWidget):
    def __init__(self, side: str, height: int):
        super().__init__()
        self.side = side
        self.setFixedWidth(14)
        self.setFixedHeight(height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 2))

        w = self.width()
        h = self.height()
        pad = 4

        if self.side == "left":
            painter.drawLine(w-2, pad, 2, pad)
            painter.drawLine(2, pad, 2, h-pad)
            painter.drawLine(2, h-pad, w-2, h-pad)
        else:
            painter.drawLine(2, pad, w-2, pad)
            painter.drawLine(w-2, pad, w-2, h-pad)
            painter.drawLine(2, h-pad, w-2, h-pad)


# ================= MATRIX =================
class MatrixWidget(QWidget):
    EXTRA_HEIGHT = 8

    def __init__(self, size: int, readonly=False, on_last_enter=None):
        super().__init__()
        self.size = size
        self.readonly = readonly
        self.on_last_enter = on_last_enter
        self.cells = []
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setSpacing(6)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignCenter)

        grid = QGridLayout()
        grid.setSpacing(CELL_SPACING)
        grid.setContentsMargins(0, 0, 0, 0)

        for r in range(self.size):
            row = []
            for c in range(self.size):
                is_last = (r == self.size - 1 and c == self.size - 1)
                cell = MatrixCell(
                    readonly=self.readonly,
                    is_last=is_last,
                    on_last_enter=self.on_last_enter
                )
                grid.addWidget(cell, r, c)
                row.append(cell)
            self.cells.append(row)

        matrix_height = (
            self.size * CELL_H +
            (self.size - 1) * CELL_SPACING
        )

        total_height = matrix_height + self.EXTRA_HEIGHT
        self.setFixedHeight(total_height)

        outer.addWidget(BracketWidget("left", total_height))
        outer.addLayout(grid)
        outer.addWidget(BracketWidget("right", total_height))

    def get_matrix(self):
        return [[int(cell.text()) for cell in row] for row in self.cells]

    def set_matrix(self, data):
        for r in range(self.size):
            for c in range(self.size):
                self.cells[r][c].setText(str(int(data[r][c])))

    def reset(self):
        for row in self.cells:
            for cell in row:
                cell.setText("0")


# ================= MAIN WINDOW =================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Multiplication")
        self.resize(1150, 550)
        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        self.size_input.setFocus()

    def _build_ui(self):
        self.root = QVBoxLayout(self)
        self.root.setSpacing(25)

        # Rank input
        self.size_input = QLineEdit()
        self.size_input.setPlaceholderText("Rank")
        self.size_input.setAlignment(Qt.AlignCenter)
        self.size_input.setFixedSize(68, 40)
        self.size_input.setFont(QFont("Consolas", 18, QFont.Bold))
        self.size_input.setValidator(QIntValidator(1, 9))
        self.size_input.returnPressed.connect(self.create_matrices)

        self.root.addWidget(self.size_input, alignment=Qt.AlignHCenter)

        # Scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.root.addWidget(self.scroll)

        container = QWidget()
        self.matrix_row = QHBoxLayout(container)
        self.matrix_row.setSpacing(25)
        self.matrix_row.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(container)

        # Buttons
        self.button_row = QHBoxLayout()
        self.button_row.setSpacing(30)
        self.button_row.setAlignment(Qt.AlignCenter)

        self.reset_btn = QPushButton("RESET")
        self.reset_btn.setFont(QFont("Consolas", 18, QFont.Bold))
        self.reset_btn.setFixedSize(BUTTON_W, BUTTON_H)
        self.reset_btn.clicked.connect(self.reset_all)

        self.calc_btn = QPushButton("CALCULATE")
        self.calc_btn.setFont(QFont("Consolas", 18, QFont.Bold))
        self.calc_btn.setFixedSize(BUTTON_W, BUTTON_H)
        self.calc_btn.clicked.connect(self.multiply)
        self.calc_btn.setEnabled(False)

        self.calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d7ef7;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #1c63d5;
            }
            QPushButton:disabled {
                background-color: #999999;
            }
        """)

        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #dddddd;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #cccccc;
            }
        """)

        self.button_row.addWidget(self.reset_btn)
        self.button_row.addWidget(self.calc_btn)

        self.reset_btn.hide()
        self.calc_btn.hide()

        self.root.addLayout(self.button_row)

    def create_matrices(self):
        if not self.size_input.text():
            return

        n = int(self.size_input.text())

        while self.matrix_row.count():
            item = self.matrix_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.A = MatrixWidget(n)
        self.B = MatrixWidget(n, on_last_enter=self.multiply)
        self.C = MatrixWidget(n, readonly=True)

        mul = QLabel("×")
        eq = QLabel("=")
        mul.setFont(FONT_OP)
        eq.setFont(FONT_OP)

        self.matrix_row.addWidget(self.A)
        self.matrix_row.addWidget(mul)
        self.matrix_row.addWidget(self.B)
        self.matrix_row.addWidget(eq)
        self.matrix_row.addWidget(self.C)

        self.reset_btn.show()
        self.calc_btn.show()
        self.calc_btn.setEnabled(True)

    def reset_all(self):
        if hasattr(self, "A"):
            self.A.reset()
            self.B.reset()
            self.C.reset()

    def multiply(self):
        A = self.A.get_matrix()
        B = self.B.get_matrix()
        n = len(A)

        result = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += A[i][k] * B[k][j]

        self.C.set_matrix(result)


# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())