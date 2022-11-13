from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import time
from dashboard import dashboard_ui
import pyqtgraph as pg

from collections import deque
from random import randint
import numpy as np
import onnxruntime
from collections import namedtuple

from audiochains.streams import InputStream, StreamFromFile
from audiochains.block_methods import UnpackRawInInt16, UnpackRawInFloat32
from audiochains.threads import AudioInThread

PhoneLabel = namedtuple('PhoneLabel', 'phone time_point')

vowel_labels = ['IY', 'IH', 'EH', 'EY', 'AE', 'AA', 'AW', 'AY', 'AH', 'AO', 'OY', 'OW', 'UH', 'UW', 'UX', 'ER', 'AX',
                'IX', 'AXR', 'AH-H']
consonant_labels = ['B', 'D', 'G', 'P', 'T', 'K', 'DX', 'Q', 'JH', 'CH', 'S', 'SH', 'Z', 'ZH', 'F', 'TH', 'V', 'M', 'N',
                    'NG', 'EM', 'EN', 'ENG', 'NX']

other_labels = ['H#', 'PAU', 'EPI']

filepath = '../realtime/M3 with other_no resample, chunk_size=1024, kernel=80, stride=4, n_channels=256, optimizer=adadelta, lr=0.03'

ort_session = onnxruntime.InferenceSession(filepath)


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def index_to_label(index):
    lbs = ['consonants', 'vowels', 'other']
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return lbs[index]


# %%
def predict(mdl, audio):
    pred = mdl(audio)
    pred = get_likely_index(pred)
    pred = index_to_label(pred)
    return pred


def predict_onnx(audio):
    # audio = audio.to_numpy().astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: audio}
    pred = ort_session.run(None, ort_inputs)
    pred = pred[0].argmax()
    pred = index_to_label(pred)
    return pred


def get_target(phone: str):
    if phone in vowel_labels:
        return 'vowels'
    elif phone in consonant_labels:
        return 'consonants'
    else:
        return 'other'


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
        self.accuracyView.setYRange(max=100.0, min=0.0)
        self.timeConsumingAudi.setYRange(max=1.0, min=0.0)

        self.startButton.clicked.connect(self.play)
        timit_constant = 15987

        self.labels = list()

        with open('../data/SA2.PHN', 'r') as file:
            for line in file:
                line = line.split()
                mark = line[2].upper()
                time_point = round(int(line[1]) / timit_constant, 3)
                label = PhoneLabel(mark, time_point)
                self.labels.append(label)

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

        with StreamFromFile(
                filename='../data/SA2.WAV.wav',
                blocksize=1024,
        ) as stream:
            self.stream = stream
            self.stream.set_methods(
                UnpackRawInFloat32()
            )

            current_label = self.labels[0]
            label_counter = 0
            iter_counter = 0

            count_correct = 0
            count_incorrect = 0

            acc_x = list()
            acc_y = list()
            time_cons_y = list()

            while not self.audio_thread.is_stopped():

                chunk = self.stream.apply()
                self.y_axes = chunk

                st = time.time()
                preds = predict_onnx(chunk.reshape(1, 1, -1))
                target = get_target(current_label.phone)
                et = time.time()

                time_cons_y.append(et - st)


                if preds == target:
                    count_correct += 1
                else:
                    count_incorrect += 1

                # seconds = (iter_counter * 1024) / 16000

                if len(self.y_axes) != 1024:
                    self.x_axes = [i for i in range(len(self.y_axes))]

                iter_counter += 1
                self.rawAudioView.data_line.setData(self.x_axes, self.y_axes)

                # print(count_correct / (count_correct + count_incorrect) * 100, iter_counter)
                acc_y.append(count_correct / (count_correct + count_incorrect) * 100)
                acc_x.append(iter_counter)

                self.accuracyView.data_line.setData(acc_x, acc_y)
                self.timeConsumingAudi.data_line.setData(acc_x, time_cons_y)

                time.sleep(0.064)

            self.stop()
            self.stream.close()


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
