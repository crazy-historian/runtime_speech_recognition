import typing

import matplotlib

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QWidget
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class PyQtGraphWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0), width=2.0)
        self.data_line = self.plot([], [], pen=pen)
        self.setYRange(0, 90.0)
        # self.setXRange(0, 2.0)

    def setup(self):
        self.setBackground('w')
        pen = pg.mkPen(color=(255, 0, 0), width=2.0)
        self.data_line = self.plot([], [], pen=pen)


class CustomMessageBox(QMessageBox):
    def __init__(self, msg_type, style_sheet="color: rgb(255, 255, 255);\n"
                                             "background-color: rgb(0, 50, 100);\n"
                                             "font: 57 16pt \"Montserrat Medium\";"):
        super().__init__()
        self.setIcon(msg_type)
        self.setStyleSheet(style_sheet)

    def critical(self,
                 parent: typing.Optional[QWidget],
                 title: str,
                 text: str,
                 buttons: typing.Union['QMessageBox.StandardButtons',
                                       'QMessageBox.StandardButton'] = ...,
                 defaultButton: 'QMessageBox.StandardButton' = ...) -> 'QMessageBox.StandardButton':
        self.setWindowTitle(title)
        self.setText(text)
        self.show()
        returned_value = self.exec_()
        return returned_value

    def question(self,
                 parent: typing.Optional[QWidget],
                 title: str,
                 text: str,
                 buttons: typing.Union['QMessageBox.StandardButtons', 'QMessageBox.StandardButton'] = ...,
                 defaultButton: 'QMessageBox.StandardButton' = ...) -> 'QMessageBox.StandardButton':
        self.setWindowTitle(title)
        self.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        self.setText(text)
        self.show()
        returned_value = self.exec_()
        return returned_value
