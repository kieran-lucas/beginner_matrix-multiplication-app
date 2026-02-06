import sys
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("PySide6 - Step 1")

window.resize(800, 500)
window.show()

sys.exit(app.exec())
