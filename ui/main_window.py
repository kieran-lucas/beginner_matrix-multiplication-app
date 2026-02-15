from PySide6.QtWidgets import QMainWindow
from ui.main_widget import MainWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Matrix Multiplication")
        self.setMinimumSize(600, 400)

        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)