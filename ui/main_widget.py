from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel,
    QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt
from core.matrix import multiply  # nếu bạn dùng function


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.main_layout = QVBoxLayout(self)

        self.inputsA = []
        self.inputsB = []
        self.resultInputs = []

        self.setup_dimension_input()

        self.matrix_layout = QHBoxLayout()
        self.main_layout.addLayout(self.matrix_layout)

        self.setup_buttons()

    # =========================
    # Dimension Input
    # =========================
    def setup_dimension_input(self):
        layout = QHBoxLayout()

        label = QLabel("Dimension (1-9):")
        self.dimension_input = QLineEdit()
        self.dimension_input.setFixedWidth(50)

        self.dimension_input.returnPressed.connect(self.create_matrices)

        layout.addWidget(label)
        layout.addWidget(self.dimension_input)
        layout.addStretch()

        self.main_layout.addLayout(layout)

    # =========================
    # Create Matrices
    # =========================
    def create_matrices(self):
        text = self.dimension_input.text()

        if not text.isdigit():
            return

        n = int(text)

        if not (1 <= n <= 9):
            return

        self.clear_layout(self.matrix_layout)

        self.inputsA = []
        self.inputsB = []
        self.resultInputs = []

        self.matrix_layout.addLayout(self.create_matrix("A", n, self.inputsA))
        self.matrix_layout.addLayout(self.create_matrix("B", n, self.inputsB))
        self.matrix_layout.addLayout(self.create_matrix("Result", n, self.resultInputs, readonly=True))

    # =========================
    # Matrix Grid Creator
    # =========================
    def create_matrix(self, title, n, container, readonly=False):
        layout = QVBoxLayout()
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)

        grid = QGridLayout()

        for i in range(n):
            row = []
            for j in range(n):
                line = QLineEdit("0")
                line.setFixedWidth(40)

                if readonly:
                    line.setReadOnly(True)
                else:
                    line.focusInEvent = self.make_focus_handler(line)

                grid.addWidget(line, i, j)
                row.append(line)
            container.append(row)

        layout.addWidget(label)
        layout.addLayout(grid)
        return layout

    # =========================
    # Remove zero when click
    # =========================
    def make_focus_handler(self, widget):
        def handler(event):
            if widget.text() == "0":
                widget.clear()
            QLineEdit.focusInEvent(widget, event)
        return handler

    # =========================
    # Buttons
    # =========================
    def setup_buttons(self):
        layout = QHBoxLayout()

        self.calculate_btn = QPushButton("Calculate")
        self.reset_btn = QPushButton("Reset")

        self.calculate_btn.clicked.connect(self.calculate)
        self.reset_btn.clicked.connect(self.reset)

        layout.addStretch()
        layout.addWidget(self.calculate_btn)
        layout.addWidget(self.reset_btn)

        self.main_layout.addLayout(layout)

    # =========================
    # Calculate
    # =========================
    def calculate(self):
        if not self.inputsA:
            return

        try:
            dataA = self.get_matrix_data(self.inputsA)
            dataB = self.get_matrix_data(self.inputsB)

            result = multiply(dataA, dataB)

            for i in range(len(result)):
                for j in range(len(result)):
                    self.resultInputs[i][j].setText(str(result[i][j]))

        except Exception:
            QMessageBox.warning(self, "Error", "Invalid input")

    # =========================
    # Get Data From UI
    # =========================
    def get_matrix_data(self, inputs):
        n = len(inputs)
        data = []

        for i in range(n):
            row = []
            for j in range(n):
                text = inputs[i][j].text()
                if text == "":
                    value = 0
                else:
                    value = int(text)
                row.append(value)
            data.append(row)

        return data

    # =========================
    # Reset
    # =========================
    def reset(self):
        self.dimension_input.clear()
        self.clear_layout(self.matrix_layout)
        self.inputsA = []
        self.inputsB = []
        self.resultInputs = []

    # =========================
    # Clear Layout Helper
    # =========================
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())