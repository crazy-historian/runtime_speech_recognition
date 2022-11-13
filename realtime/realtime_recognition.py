import numpy as np
import onnxruntime

import torch
from torch import nn
from collections import namedtuple
import torch.nn.functional as F
import sounddevice
from audiochains.streams import InputStream, StreamFromFile
from audiochains.block_methods import UnpackRawInInt16, UnpackRawInFloat32
from models.phoneme_recognizer import PhonemeRecognizer

PhoneLabel = namedtuple('PhoneLabel', 'phone time_point')

vowel_labels = ['IY', 'IH', 'EH', 'EY', 'AE', 'AA', 'AW', 'AY', 'AH', 'AO', 'OY', 'OW', 'UH', 'UW', 'UX', 'ER', 'AX',
                'IX', 'AXR', 'AH-H']
consonant_labels = ['B', 'D', 'G', 'P', 'T', 'K', 'DX', 'Q', 'JH', 'CH', 'S', 'SH', 'Z', 'ZH', 'F', 'TH', 'V', 'M', 'N',
                    'NG', 'EM', 'EN', 'ENG', 'NX']

other_labels = ['H#', 'PAU', 'EPI']

filepath = 'M3 with other_no resample, chunk_size=1024, kernel=80, stride=4, n_channels=256, optimizer=adadelta, lr=0.03'

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


class M3(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=4, kernel_size=80, n_channel=256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        #
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = F.avg_pool1d(x, int(x.shape[-1]))
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
        # return self.sigmoid(x)


model = PhonemeRecognizer.load_from_checkpoint(
    'ckeckpoint.ckpt'
)

if __name__ == "__main__":
    with StreamFromFile(
            filename='../data/SA2.WAV.wav',
            blocksize=1024,
    ) as stream, open('../data/SA2.PHN', 'r') as file:

        timit_constant = 15987
        labels = list()

        for line in file:
            line = line.split()
            mark = line[2].upper()
            time_point = round(int(line[1]) / timit_constant, 3)
            label = PhoneLabel(mark, time_point)
            labels.append(label)

        current_label = labels[0]
        label_counter = 0
        iter_counter = 0

        count_correct = 0
        count_incorrect = 0

        stream.set_methods(
            UnpackRawInFloat32()
        )

        print(iter_counter, 0, current_label)
        for _ in range(stream.get_iterations()):
            chunk = stream.apply()
            # chunk = torch.from_numpy(chunk.copy())

            preds = predict_onnx(chunk.reshape(1, 1, -1))
            target = get_target(current_label.phone)

            if preds == target:
                correct = True
                count_correct += 1
            else:
                count_incorrect += 1
                correct = False

            seconds = (iter_counter * 1024) / 16000
            # print(iter_counter, seconds, current_label.phone, preds, correct)

            if seconds > current_label.time_point:
                label_counter += 1
                current_label = labels[label_counter]

            iter_counter += 1

            print(f'Total accuracy: {count_correct / (count_correct + count_incorrect) * 100}')
