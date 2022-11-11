from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
from dashboard import dashboard_ui
import pyqtgraph as pg

from collections import deque
from random import randint
import numpy as np

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInInt16, UnpackRawInFloat32
from audiochains.threads import AudioInThread


class ExampleApp(QtWidgets.QMainWindow, dashboard_ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)

        self.stream = None
        self.audio_thread = None

        self.button_pressed = False

        self.x_axes = [i for i in range(1024)]
        self.y_axes = list()
        self.rawAudioView.setYRange(max=1.0, min=-1.0)
        self.rawAudioView.setYRange(max=1.0, min=-1.0)
        self.startButton.clicked.connect(self.play)

    def play(self):
        if self.button_pressed is False:
            self.button_pressed = True
            self.x_axes = [i for i in range(1024)]
            self.y_axes = list()
            self.audio_thread = AudioInThread(target=self.update_plot_data)
            self.audio_thread.start()
        else:
            self.button_pressed = False
            self.stop()

    def stop(self):
        if self.audio_thread:
            self.audio_thread.stop()

    def update_plot_data(self):
        self.stream = InputStream(
            samplerate=16000,
            blocksize=1024,
            channels=1,
            sampwidth=2)
        self.stream.set_methods(
            UnpackRawInFloat32()
        )
        self.stream.start()
        while not self.audio_thread.is_stopped():
            self.y_axes = self.stream.apply()
            self.rawAudioView.data_line.setData(self.x_axes, self.y_axes)

        self.stop()
        self.stream.close()


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
