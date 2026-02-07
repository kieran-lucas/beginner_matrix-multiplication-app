import sys
import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass
from PySide6.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton, QVBoxLayout, 
    QHBoxLayout, QLabel, QGridLayout, QFrame, QGraphicsDropShadowEffect,
    QTabWidget, QSpinBox, QComboBox, QTextEdit, QSplitter, QGroupBox,
    QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMenu, QSizePolicy, QStackedWidget, QCheckBox, QScrollArea
)
from PySide6.QtCore import (
    Qt, QLocale, QThread, Signal, QObject, QTimer, 
    QEasingCurve, QPropertyAnimation, QParallelAnimationGroup,
    QSize, QRunnable, QThreadPool, QEvent
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QDoubleValidator, QFont, 
    QFontMetrics, QLinearGradient, QBrush, QAction,
    QPainterPath, QKeyEvent, QPalette, QSyntaxHighlighter,
    QTextCharFormat
)
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from datetime import datetime

# ===== CONSTANTS & CONFIGURATION =====
class Config:
    # Colors
    COLOR_BG = "#0A0E14"
    COLOR_CARD = "#151A21"
    COLOR_ACCENT = "#3794FF"
    COLOR_SECONDARY = "#FF6B9D"
    COLOR_SUCCESS = "#4EC9B0"
    COLOR_WARNING = "#DCDCAA"
    COLOR_ERROR = "#F44747"
    COLOR_TEXT = "#D4D4D4"
    COLOR_SUBTEXT = "#858585"
    
    # Sizes
    MAX_MATRIX_SIZE = 8
    MIN_MATRIX_SIZE = 2
    DEFAULT_MATRIX_SIZE = 3
    ANIMATION_DURATION = 300
    
    # Computation
    MAX_WORKERS = 4
    DECIMALS = 4
    PRECISION = 1e-10
    
    # Cache
    CACHE_SIZE = 100

# ===== ADVANCED MATRIX OPERATIONS =====
class MatrixOperations:
    @staticmethod
    def strassen_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Strassen algorithm for matrix multiplication (O(n^2.81))"""
        n = len(A)
        if n <= 64:  # Fallback to standard multiplication for small matrices
            return np.dot(A, B)
        
        # Split matrices into quarters
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]
        
        # Calculate 7 products recursively
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(MatrixOperations.strassen_multiply, A11 + A22, B11 + B22),
                executor.submit(MatrixOperations.strassen_multiply, A21 + A22, B11),
                executor.submit(MatrixOperations.strassen_multiply, A11, B12 - B22),
                executor.submit(MatrixOperations.strassen_multiply, A22, B21 - B11),
                executor.submit(MatrixOperations.strassen_multiply, A11 + A12, B22),
                executor.submit(MatrixOperations.strassen_multiply, A21 - A11, B11 + B12),
                executor.submit(MatrixOperations.strassen_multiply, A12 - A22, B21 + B22)
            ]
            M = [f.result() for f in futures]
        
        # Combine results
        C11 = M[0] + M[3] - M[4] + M[6]
        C12 = M[2] + M[4]
        C21 = M[1] + M[3]
        C22 = M[0] - M[1] + M[2] + M[5]
        
        return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    
    @staticmethod
    def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LU decomposition with partial pivoting"""
        n = len(A)
        L = np.eye(n)
        U = A.copy()
        P = np.eye(n)
        
        for i in range(n):
            # Pivot
            max_row = np.argmax(np.abs(U[i:, i])) + i
            if max_row != i:
                U[[i, max_row]] = U[[max_row, i]]
                P[[i, max_row]] = P[[max_row, i]]
                if i > 0:
                    L[[i, max_row], :i] = L[[max_row, i], :i]
            
            # Elimination
            for j in range(i + 1, n):
                L[j, i] = U[j, i] / U[i, i]
                U[j, i:] -= L[j, i] * U[i, i:]
        
        return P, L, U
    
    @staticmethod
    def eigenvalue_power_method(A: np.ndarray, iterations: int = 1000) -> Tuple[float, np.ndarray]:
        """Power method for dominant eigenvalue/vector"""
        n = len(A)
        b = np.random.rand(n)
        b = b / np.linalg.norm(b)
        
        for _ in range(iterations):
            Ab = np.dot(A, b)
            eigenvalue = np.dot(b, Ab)
            b_new = Ab / np.linalg.norm(Ab)
            
            if np.linalg.norm(b_new - b) < Config.PRECISION:
                break
            b = b_new
        
        return float(eigenvalue), b
    
    @staticmethod
    def matrix_exponential(A: np.ndarray, terms: int = 20) -> np.ndarray:
        """Matrix exponential using Taylor series"""
        n = len(A)
        result = np.eye(n)
        Ak = np.eye(n)
        
        for k in range(1, terms + 1):
            Ak = np.dot(Ak, A) / k
            result += Ak
            
            if np.linalg.norm(Ak) < Config.PRECISION:
                break
        
        return result

# ===== COMPUTATION WORKER =====
class MatrixWorkerSignals(QObject):
    progress = Signal(int)
    result_ready = Signal(object, str)
    error = Signal(str)
    started = Signal()
    finished = Signal()

class MatrixWorker(QRunnable):
    def __init__(self, operation: str, matrices: Dict[str, np.ndarray], **kwargs):
        super().__init__()
        self.operation = operation
        self.matrices = matrices
        self.kwargs = kwargs
        self.signals = MatrixWorkerSignals()
        self.is_cancelled = False
    
    def run(self):
        try:
            self.signals.started.emit()
            
            if self.operation == "multiply":
                A, B = self.matrices['A'], self.matrices['B']
                use_strassen = self.kwargs.get('use_strassen', False)
                
                if use_strassen and len(A) >= 4 and len(A) & (len(A)-1) == 0:
                    result = MatrixOperations.strassen_multiply(A, B)
                else:
                    result = np.dot(A, B)
                    
            elif self.operation == "lu":
                P, L, U = MatrixOperations.lu_decomposition(self.matrices['A'])
                result = {'P': P, 'L': L, 'U': U}
                
            elif self.operation == "eigenvalues":
                eigenvalues, eigenvectors = np.linalg.eig(self.matrices['A'])
                result = {
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors
                }
                
            elif self.operation == "inverse":
                result = np.linalg.inv(self.matrices['A'])
                
            elif self.operation == "determinant":
                result = np.linalg.det(self.matrices['A'])
                
            elif self.operation == "exponential":
                result = MatrixOperations.matrix_exponential(self.matrices['A'])
                
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
            
            if not self.is_cancelled:
                self.signals.result_ready.emit(result, self.operation)
                
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
    
    def cancel(self):
        self.is_cancelled = True

# ===== CACHE SYSTEM =====
class MatrixCache:
    def __init__(self, max_size: int = Config.CACHE_SIZE):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get_key(self, operation: str, matrices: tuple) -> str:
        """Generate cache key from operation and matrix data"""
        import hashlib
        data = f"{operation}_{matrices}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get(self, key: str):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value):
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)

# ===== ADVANCED MATRIX INPUT =====
class MatrixSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(Config.COLOR_SUCCESS))
        self.highlighting_rules.append((r'\b\d+\.?\d*\b', number_format))
        
        # Operators
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor(Config.COLOR_ACCENT))
        operators = r'[+\-*/=(),\[\]]'
        self.highlighting_rules.append((operators, operator_format))
        
        # Matrix identifiers
        matrix_format = QTextCharFormat()
        matrix_format.setForeground(QColor(Config.COLOR_SECONDARY))
        self.highlighting_rules.append((r'\b[A-Z]\b', matrix_format))
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            import re
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), format)

class MatrixInput(QWidget):
    def __init__(self, rows: int, cols: int, label: str = "", readonly: bool = False, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.label = label
        self.readonly = readonly
        self.inputs = []
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        if label:
            title = QLabel(label)
            title.setObjectName("MatrixTitle")
            self.layout.addWidget(title)
        
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(6)
        
        validator = QDoubleValidator(-1e12, 1e12, Config.DECIMALS, self)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        validator.setLocale(QLocale.c())
        
        for i in range(rows):
            row = []
            for j in range(cols):
                box = QLineEdit("0")
                box.setFixedSize(48, 40)
                box.setAlignment(Qt.AlignCenter)
                box.setValidator(validator)
                box.setObjectName("MatrixCell")
                box.setReadOnly(readonly)
                box.textChanged.connect(self.on_cell_changed)
                grid.addWidget(box, i, j)
                row.append(box)
            self.inputs.append(row)
        
        self.layout.addWidget(grid_widget)
        self.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Tab, Qt.Key_Backtab):
                self.navigate_cells(event.key() == Qt.Key_Tab)
                return True
        return super().eventFilter(obj, event)
    
    def navigate_cells(self, forward: bool):
        current = QApplication.focusWidget()
        if not isinstance(current, QLineEdit):
            return
        
        all_cells = [cell for row in self.inputs for cell in row]
        if current not in all_cells:
            return
        
        idx = all_cells.index(current)
        if forward:
            next_idx = (idx + 1) % len(all_cells)
        else:
            next_idx = (idx - 1) % len(all_cells)
        
        all_cells[next_idx].setFocus()
        all_cells[next_idx].selectAll()
    
    def on_cell_changed(self):
        sender = self.sender()
        if sender and sender.text():
            try:
                value = float(sender.text())
                sender.setStyleSheet("")
            except:
                sender.setStyleSheet("background-color: rgba(244, 71, 71, 0.3);")
    
    def get_matrix(self) -> np.ndarray:
        return np.array([
            [float(cell.text() or 0) for cell in row]
            for row in self.inputs
        ])
    
    def set_matrix(self, matrix: np.ndarray):
        for i in range(min(self.rows, len(matrix))):
            for j in range(min(self.cols, len(matrix[0]))):
                value = matrix[i][j]
                text = f"{value:.{Config.DECIMALS}g}"
                self.inputs[i][j].setText(text)
    
    def randomize(self):
        import random
        matrix = np.random.randn(self.rows, self.cols)
        self.set_matrix(matrix)
    
    def clear(self):
        for row in self.inputs:
            for cell in row:
                cell.setText("0")

# ===== MATRIX VISUALIZATION =====
class MatrixVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix = None
        self.color_map = None
        self.setMinimumSize(200, 200)
    
    def set_matrix(self, matrix: np.ndarray):
        self.matrix = matrix
        self.normalize_matrix()
        self.update()
    
    def normalize_matrix(self):
        if self.matrix is None:
            return
        
        abs_matrix = np.abs(self.matrix)
        if abs_matrix.max() > 0:
            self.color_map = abs_matrix / abs_matrix.max()
        else:
            self.color_map = np.zeros_like(self.matrix)
    
    def paintEvent(self, event):
        if self.matrix is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rows, cols = self.matrix.shape
        cell_width = self.width() / cols
        cell_height = self.height() / rows
        
        for i in range(rows):
            for j in range(cols):
                value = self.matrix[i][j]
                intensity = self.color_map[i][j]
                
                # Color based on value
                if value >= 0:
                    color = QColor(
                        int(55 + 200 * intensity),
                        int(148 + 107 * intensity),
                        int(255)
                    )
                else:
                    color = QColor(
                        int(255),
                        int(107 + 148 * intensity),
                        int(55 + 200 * intensity)
                    )
                
                painter.fillRect(
                    int(j * cell_width),
                    int(i * cell_height),
                    int(cell_width),
                    int(cell_height),
                    color
                )
                
                # Draw value
                painter.setPen(QColor(Config.COLOR_TEXT))
                text = f"{value:.2f}"
                painter.drawText(
                    int(j * cell_width + cell_width/2 - 15),
                    int(i * cell_height + cell_height/2 + 4),
                    text
                )

# ===== MAIN OPERATION CARD =====
class MatrixOperationCard(QFrame):
    def __init__(self, size: int, operation: str = "multiply", parent=None):
        super().__init__(parent)
        self.size = size
        self.operation = operation
        self.worker = None
        self.cache = MatrixCache()
        
        self.setup_ui()
        self.setup_animations()
    
    def setup_ui(self):
        self.setObjectName("Card")
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Header
        header = QHBoxLayout()
        title = QLabel(self.get_operation_title())
        title.setObjectName("CardTitle")
        
        self.size_spin = QSpinBox()
        self.size_spin.setRange(Config.MIN_MATRIX_SIZE, Config.MAX_MATRIX_SIZE)
        self.size_spin.setValue(self.size)
        self.size_spin.valueChanged.connect(self.resize_matrices)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("Size:"))
        header.addWidget(self.size_spin)
        layout.addLayout(header)
        
        # Matrix inputs
        self.matrix_widget = QWidget()
        matrix_layout = QHBoxLayout(self.matrix_widget)
        matrix_layout.setSpacing(20)
        
        self.A = MatrixInput(self.size, self.size, "Matrix A")
        self.B = MatrixInput(self.size, self.size, "Matrix B")
        self.C = MatrixInput(self.size, self.size, "Result", True)
        
        matrix_layout.addWidget(self.A)
        matrix_layout.addWidget(QLabel(self.get_operation_symbol(), objectName="MathSymbol"))
        matrix_layout.addWidget(self.B)
        matrix_layout.addWidget(QLabel("=", objectName="MathSymbol"))
        matrix_layout.addWidget(self.C)
        
        layout.addWidget(self.matrix_widget)
        
        # Visualization
        self.viz_widget = MatrixVisualization()
        self.viz_widget.setMaximumHeight(200)
        layout.addWidget(self.viz_widget)
        
        # Controls
        controls = QHBoxLayout()
        
        self.calc_btn = QPushButton("▶ CALCULATE")
        self.calc_btn.clicked.connect(self.calculate)
        
        self.random_btn = QPushButton("🎲 RANDOM")
        self.random_btn.clicked.connect(self.randomize)
        
        self.clear_btn = QPushButton("🗑️ CLEAR")
        self.clear_btn.clicked.connect(self.clear)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        controls.addWidget(self.calc_btn)
        controls.addWidget(self.random_btn)
        controls.addWidget(self.clear_btn)
        controls.addWidget(self.progress)
        
        layout.addLayout(controls)
        
        # Advanced options
        advanced = QGroupBox("Advanced Options")
        advanced_layout = QHBoxLayout(advanced)
        
        self.use_strassen = QCheckBox("Use Strassen Algorithm")
        self.parallel_check = QCheckBox("Parallel Computation")
        self.cache_check = QCheckBox("Enable Cache")
        
        advanced_layout.addWidget(self.use_strassen)
        advanced_layout.addWidget(self.parallel_check)
        advanced_layout.addWidget(self.cache_check)
        
        layout.addWidget(advanced)
    
    def setup_animations(self):
        self.animation_group = QParallelAnimationGroup()
        
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(Config.ANIMATION_DURATION)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.7)
        self.fade_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(Config.ANIMATION_DURATION)
        
        self.animation_group.addAnimation(self.fade_animation)
        self.animation_group.addAnimation(self.scale_animation)
    
    def get_operation_title(self) -> str:
        titles = {
            "multiply": "Matrix Multiplication",
            "lu": "LU Decomposition",
            "eigenvalues": "Eigen Analysis",
            "inverse": "Matrix Inverse",
            "determinant": "Determinant"
        }
        return titles.get(self.operation, "Matrix Operation")
    
    def get_operation_symbol(self) -> str:
        symbols = {
            "multiply": "×",
            "lu": "→ LU",
            "inverse": "⁻¹",
            "determinant": "det"
        }
        return symbols.get(self.operation, "→")
    
    def resize_matrices(self, new_size: int):
        self.size = new_size
        # Note: In a full implementation, you would need to recreate the matrix inputs
        # For now, we'll just update the size variable
    
    def calculate(self):
        try:
            A = self.A.get_matrix()
            B = self.B.get_matrix()
            
            # Check cache
            if self.cache_check.isChecked():
                cache_key = self.cache.get_key(
                    self.operation,
                    (A.tobytes(), B.tobytes())
                )
                cached = self.cache.get(cache_key)
                if cached is not None:
                    self.C.set_matrix(cached)
                    self.viz_widget.set_matrix(cached)
                    return
            
            # Setup worker
            self.calc_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # Indeterminate
            
            self.worker = MatrixWorker(
                self.operation,
                {'A': A, 'B': B},
                use_strassen=self.use_strassen.isChecked()
            )
            self.worker.signals.result_ready.connect(self.on_result_ready)
            self.worker.signals.error.connect(self.on_error)
            self.worker.signals.finished.connect(self.on_finished)
            
            QThreadPool.globalInstance().start(self.worker)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def on_result_ready(self, result, operation):
        if isinstance(result, np.ndarray):
            self.C.set_matrix(result)
            self.viz_widget.set_matrix(result)
            
            # Cache result
            if self.cache_check.isChecked():
                cache_key = self.cache.get_key(
                    operation,
                    (self.A.get_matrix().tobytes(), 
                     self.B.get_matrix().tobytes())
                )
                self.cache.set(cache_key, result)
        elif isinstance(result, dict):
            # Handle dictionary results (e.g., LU decomposition)
            pass
    
    def on_error(self, error_msg):
        QMessageBox.critical(self, "Computation Error", error_msg)
    
    def on_finished(self):
        self.calc_btn.setEnabled(True)
        self.progress.setVisible(False)
    
    def randomize(self):
        self.A.randomize()
        self.B.randomize()
    
    def clear(self):
        self.A.clear()
        self.B.clear()
        self.C.clear()
        self.viz_widget.set_matrix(np.zeros((self.size, self.size)))

# ===== MAIN WINDOW =====
class ModernMatrixApp(QWidget):
    def __init__(self):
        super().__init__()
        self.cache = MatrixCache()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(Config.MAX_WORKERS)
        
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Matrix Lab Pro")
        self.resize(1400, 900)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("🧮 MATRIX LABORATORY")
        title.setObjectName("AppTitle")
        
        subtitle = QLabel("Advanced Linear Algebra Computation Suite")
        subtitle.setObjectName("AppSubtitle")
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(subtitle)
        
        main_layout.addLayout(header)
        
        # Simple version - single card
        card = MatrixOperationCard(3, "multiply")
        main_layout.addWidget(card)
        
        # Footer
        footer = QHBoxLayout()
        
        stats_label = QLabel("Ready")
        stats_label.setObjectName("StatsLabel")
        
        self.memory_label = QLabel("Cache: 0/100")
        self.memory_label.setObjectName("MemoryLabel")
        
        footer.addWidget(stats_label)
        footer.addStretch()
        footer.addWidget(self.memory_label)
        
        main_layout.addLayout(footer)

# ===== ENHANCED STYLESHEET =====
QSS = f"""
/* Main Theme */
QWidget {{
    background-color: {Config.COLOR_BG};
    color: {Config.COLOR_TEXT};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}

/* App Title */
#AppTitle {{
    font-size: 28px;
    font-weight: 800;
    color: {Config.COLOR_ACCENT};
    letter-spacing: 1px;
}}

#AppSubtitle {{
    color: {Config.COLOR_SUBTEXT};
    font-size: 14px;
    font-weight: 300;
}}

/* Cards */
#Card {{
    background-color: {Config.COLOR_CARD};
    border-radius: 16px;
    border: 1px solid #2A2F3A;
    padding: 0px;
}}

#Card:hover {{
    border-color: {Config.COLOR_ACCENT}40;
}}

#CardTitle {{
    font-size: 16px;
    font-weight: 600;
    color: {Config.COLOR_TEXT};
}}

/* Matrix Cells */
#MatrixCell {{
    background: #1A1F29;
    border: 1px solid #2A2F3A;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', monospace;
    font-size: 13px;
}}

#MatrixCell:focus {{
    border: 2px solid {Config.COLOR_ACCENT};
    background: #232834;
}}

#MatrixCell:read-only {{
    background: #151A21;
    border-color: {Config.COLOR_SUCCESS}40;
    color: {Config.COLOR_SUCCESS};
}}

/* Buttons */
QPushButton {{
    background-color: {Config.COLOR_ACCENT};
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    min-width: 100px;
}}

QPushButton:hover {{
    background-color: #4CA6FF;
}}

QPushButton:pressed {{
    background-color: #1A7FFF;
}}

QPushButton:disabled {{
    background-color: #2A2F3A;
    color: {Config.COLOR_SUBTEXT};
}}

/* Progress Bar */
QProgressBar {{
    border: 1px solid #2A2F3A;
    border-radius: 4px;
    background: #151A21;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {Config.COLOR_ACCENT};
    border-radius: 4px;
}}

/* Math Symbols */
#MathSymbol {{
    font-size: 32px;
    color: {Config.COLOR_ACCENT};
    font-weight: 300;
    padding: 0 10px;
}}

/* Group Box */
QGroupBox {{
    border: 1px solid #2A2F3A;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 500;
    color: {Config.COLOR_SUBTEXT};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}}

/* Checkbox */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid #2A2F3A;
    border-radius: 4px;
}}

QCheckBox::indicator:checked {{
    background: {Config.COLOR_ACCENT};
    border-color: {Config.COLOR_ACCENT};
}}

/* Spin Box */
QSpinBox {{
    background: #1A1F29;
    border: 1px solid #2A2F3A;
    border-radius: 6px;
    padding: 5px;
    min-width: 60px;
}}

/* Status Labels */
#StatsLabel {{
    color: {Config.COLOR_SUBTEXT};
    font-size: 11px;
    font-style: italic;
}}

#MemoryLabel {{
    color: {Config.COLOR_WARNING};
    font-size: 11px;
    font-weight: 500;
    background: #2A2F3A;
    padding: 4px 8px;
    border-radius: 4px;
}}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ModernMatrixApp()
    window.show()
    
    sys.exit(app.exec())