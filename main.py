

import sys
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Test")
window.resize(800, 500)
window.show()

sys.exit(app.exec())








