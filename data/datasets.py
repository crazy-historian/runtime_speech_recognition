import pandas as pd
import numpy as np
import textgrid
import torchaudio
import torch
from pathlib import Path
from pydub import AudioSegment
from torch.utils.data import Dataset

import torch.nn.functional as F
from dataclasses import dataclass, astuple
from typing import Optional, Union, List, Callable


@dataclass(frozen=True)
class AudioFragment:
    source_file: str
    label: str
    t1: float
    t2: float


@dataclass
class AudioData:
    data: Union[bytes, np.ndarray, torch.Tensor]
    label: str
    frame_rate: int
    sample_width: int

    def __iter__(self):
        return iter(astuple(self))


class TIMITDataset(Dataset):
    """ The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus """

    def __init__(self,
                 root_dir: str,
                 usage: str,
                 padding: int = 0,
                 percentage: Optional[float] = None,
                 transform: Optional[Callable] = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 dialect: Optional[List[str]] = None
                 ):
        self.padding = padding
        self.timit_constant = 15987
        self.root_dir = root_dir
        self.phone_codes = phone_codes
        self.transform = transform
        self.description_table = self._prepare_description()
        self.description_table = self._filter_description_table(usage, percentage, gender, dialect)
        self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self):
        if Path('./data/timit_description.csv').is_file():
            return pd.read_csv('./data/timit_description.csv')
        else:
            dialects = {'DR1': 'New England', 'DR2': 'Northern', 'DR3': 'North Midland', 'DR4': 'South Midland',
                        'DR5': 'Southern', 'DR6': 'New York City', 'DR7': 'Western', 'DR8': 'Army Brat'}
            test_dir = Path(self.root_dir, 'TEST')
            train_dir = Path(self.root_dir, 'TRAIN')

            table = list()

            for directory, usage in [(test_dir, 'test'), (train_dir, 'train')]:
                for dialect in directory.iterdir():
                    for speaker in dialect.iterdir():
                        speaker_dir_name = str(speaker.stem)
                        speaker_gender = speaker_dir_name[0]
                        speaker_id = speaker_dir_name[1:]
                        for wav_file in speaker.glob('*.WAV.wav'):
                            table.append(
                                [usage,
                                 speaker_id,
                                 speaker_gender,
                                 dialects[str(dialect.stem)],
                                 str(wav_file),
                                 str(Path(speaker, wav_file.stem.split('.')[0] + '.PHN'))]
                            )
            df = pd.DataFrame(data=table, columns=['usage', 'speaker_id', 'speaker_gender',
                                                   'dialect', 'wav_file_path', 'labels_file_path'])
            df.to_csv('./data/timit_description.csv', index=False)

            return df

    def _filter_description_table(self, usage: str, percentage: Optional[float], gender: Optional[str],
                                  dialect: Optional[List[str]]) -> pd.DataFrame:
        self.description_table = self.description_table.loc[self.description_table['usage'] == usage]

        if percentage is not None:
            self.description_table = self.description_table.sample(frac=percentage)

        if gender is not None:
            self.description_table = self.description_table.loc[self.description_table['gender'] == gender]

        if dialect is not None:
            dialects = self.description_table['dialect'].isin(dialect)
            self.description_table = self.description_table[dialects]

        return self.description_table

    def _get_audio_fragments(self) -> list:
        fragments = list()
        self.description_table = self.description_table.sample(frac=1.0)
        for _, file in self.description_table.iterrows():
            timings = list()
            with open(file['labels_file_path']) as labels:
                for label in labels:
                    label = label.split()
                    mark = label[2].upper()
                    if self.phone_codes is None or mark in self.phone_codes:
                        start = round(int(label[0]) / self.timit_constant, 3)
                        end = round(int(label[1]) / self.timit_constant, 3)
                        timings.append((mark, start, end))

                for timing in timings:
                    fragments.append(
                        AudioFragment(
                            source_file=file['wav_file_path'],
                            label=timing[0],
                            t1=timing[1],
                            t2=timing[2]
                        )
                    )
        return fragments

    def _load_audio_fragment(self, audio_fragment: AudioFragment) -> AudioData:
        metadata, encoding = torchaudio.info(audio_fragment.source_file)
        frame_rate = int(metadata.rate)
        sample_width = encoding.bits_per_sample
        t1 = round(audio_fragment.t1 * frame_rate)
        t2 = round(audio_fragment.t2 * frame_rate)
        data, _ = torchaudio.load(audio_fragment.source_file)
        data = data[:, t1:t2]
        if self.padding != 0:
            new_shape = self.padding - data.shape[1]
            return AudioData(
                data=F.pad(data, (0, new_shape), 'constant', 0.0),
                label=audio_fragment.label,
                frame_rate=frame_rate,
                sample_width=sample_width
            )
        else:
            return AudioData(
                data=data,
                label=audio_fragment.label,
                frame_rate=frame_rate,
                sample_width=sample_width
            )


    def __len__(self) -> int:
        return len(self.audio_fragments)

    def __getitem__(self, item: int) -> AudioData:
        if self.transform:
            audio_data = self._load_audio_fragment(self.audio_fragments[item])
            audio_data.data = self.transform(audio_data.data)
            return audio_data
        return self._load_audio_fragment(self.audio_fragments[item])


class ArcticDataset(Dataset):
    """ ARCTIC L2 dataset """

    def __init__(self,
                 root_dir: str,
                 usage: str,
                 padding: int = 0,
                 fraction: float = 0.7,
                 transform: Callable = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 dialect: Optional[List[str]] = None):
        self.padding = 0
        self.root_dir = root_dir
        self.transform = transform
        self.phone_codes = phone_codes
        self.speaker_description = {
            'ABA': ['Arabic', 'M'],
            'SKA': ['Arabic', 'F'],
            'YBAA': ['Arabic', 'M'],
            'ZHAA': ['Arabic', 'F'],
            'BWC': ['Mandarin', 'M'],
            'LXC': ['Mandarin', 'F'],
            'NCC': ['Mandarin', 'F'],
            'TXHC': ['Mandarin', 'M'],
            'ASI': ['Hindi', 'M'],
            'RRBI': ['Hindi', 'M'],
            'SVBI': ['Hindi', 'F'],
            'TNI': ['Hindi', 'F'],
            'HJK': ['Korean', 'F'],
            'HKK': ['Korean', 'M'],
            'YDCK': ['Korean', 'F'],
            'YKWK': ['Korean', 'M'],
            'EBVS': ['Spanish', 'M'],
            'ERMS': ['Spanish', 'M'],
            'MBMPS': ['Spanish', 'F'],
            'NJS': ['Spanish', 'F'],
            'HQTV': ['Vietnamese', 'M'],
            'PNV': ['Vietnamese', 'F'],
            'THV': ['Vietnamese', 'F'],
            'TLV': ['Vietnamese', 'M']
        }
        self.description_table = self._prepare_description(fraction)
        self.description_table = self._filter_description_table(usage, gender, dialect)
        self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self, fraction: float) -> pd.DataFrame:
        if Path('./data/arctic_description.csv').is_file():
            return pd.read_csv('./data/arctic_description.csv')
        else:
            table = list()
            for speaker_dir in Path(self.root_dir).iterdir():
                table_rows = list()
                for annotation_file in Path(speaker_dir, 'annotation').iterdir():
                    audio_dir_path = str(Path(speaker_dir, 'wav'))
                    audio_file = Path(audio_dir_path, f'{annotation_file.stem}.wav')
                    table_rows.append([
                        speaker_dir.stem,
                        self.speaker_description[speaker_dir.stem][0],
                        self.speaker_description[speaker_dir.stem][1],
                        str(annotation_file),
                        str(audio_file)
                    ])
                table.extend(table_rows)

            df = pd.DataFrame(data=table, columns=['nickname', 'l1', 'gender', 'labels_file_path', 'wav_file_path'])
            df['usage'] = 'train'
            df.loc[df.sample(frac=fraction).index.to_list(), 'usage'] = 'test'
            df.to_csv('./data/arctic_description.csv', index=False)

            return df

    def _filter_description_table(self, usage: str, gender: Optional[str], dialect: Optional[str]) -> pd.DataFrame:
        self.description_table = self.description_table.loc[self.description_table['usage'] == usage]

        if gender is not None:
            self.description_table = self.description_table.loc[self.description_table['gender'] == gender]

        if dialect is not None:
            dialects = self.description_table['l1'].isin(dialect)
            self.description_table = self.description_table[dialects]

        return self.description_table

    def _get_audio_fragments(self) -> list:
        fragments = list()
        self.description_table = self.description_table.sample(frac=1.0)
        try:
            for _, file in self.description_table.iterrows():
                labels = textgrid.TextGrid.fromFile(file['labels_file_path'])
                timings = list()
                for interval in labels[1]:
                    if self.phone_codes is None or interval.mark in self.phone_codes:
                        start = interval.minTime
                        end = interval.maxTime
                        timings.append((interval.mark, start, end))
                for timing in timings:
                    fragments.append(
                        AudioFragment(
                            source_file=file['wav_file_path'],
                            label=timing[0],
                            t1=timing[1],
                            t2=timing[2]
                        )
                    )
        except ValueError:
            ...
        return fragments

    def _load_audio_fragment(self, audio_fragment: AudioFragment) -> AudioData:
        metadata, encoding = torchaudio.info(audio_fragment.source_file)
        frame_rate = int(metadata.rate)
        sample_width = encoding.bits_per_sample
        t1 = round(audio_fragment.t1 * frame_rate)
        t2 = round(audio_fragment.t2 * frame_rate)
        data, _ = torchaudio.load(audio_fragment.source_file)
        data = data[:, t1:t2]
        if self.padding != 0:
            new_shape = self.padding - data.shape[1]
            return AudioData(
                data=F.pad(data, (0, new_shape), 'constant', 0.0),
                label=audio_fragment.label,
                frame_rate=frame_rate,
                sample_width=sample_width
            )
        else:
            return AudioData(
                data=data,
                label=audio_fragment.label,
                frame_rate=frame_rate,
                sample_width=sample_width
            )

    def __len__(self) -> int:
        return len(self.audio_fragments)

    def __getitem__(self, item: int) -> AudioData:
        if self.transform:
            audio_data = self._load_audio_fragment(self.audio_fragments[item])
            audio_data.data = self.transform(audio_data.data)
            return audio_data
        return self._load_audio_fragment(self.audio_fragments[item])
