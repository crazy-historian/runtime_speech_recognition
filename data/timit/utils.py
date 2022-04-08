import pandas as pd
from pathlib import Path
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Callable


class TIMITDataset(Dataset):
    """ The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[Callable] = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 dialect: Optional[List[str]] = None
                 ):
        self.root_dir = root_dir
        self.phone_codes = phone_codes
        self.transform = transform
        self.description_table = self._prepare_description()
        self.description_table = self._filter_description_table(gender, dialect)
        self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self):
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
                             str(Path(speaker, wav_file.stem.split('.')[0] + '.phn'))]
                        )
        df = pd.DataFrame(data=table, columns=['usage', 'speaker_id', 'speaker_gender',
                                               'dialect', 'wav_file_path', 'labels_file_path'])
        df.to_csv('timit_description.csv', index=False)

        return df

    def _filter_description_table(self, gender: Optional[str], dialect: Optional[List[str]]) -> pd.DataFrame:
        if gender is not None:
            self.description_table = self.description_table.loc[
                self.description_table['gender'] == gender
                ]
        if dialect is not None:
            dialects = self.description_table['dialect'].isin(dialect)
            self.description_table = self.description_table[dialects]

        return self.description_table

    def _get_audio_fragments(self) -> list:
        timit_constant = 0.0625
        fragments = list()
        for _, file in self.description_table.iterrows():
            timings = list()
            with open(file['labels_file_path']) as labels:
                for label in labels:
                    label = label.split()
                    mark = label[2]
                    if self.phone_codes is None or mark in self.phone_codes:
                        start = round(int(label[0]) * timit_constant)
                        end = round(int(label[1]) * timit_constant)
                        timings.append((mark, start, end))
                if len(timings) > 0:
                    for timing in timings:
                        wav_file = AudioSegment.from_wav(file['wav_file_path'])
                        fragments.append([timing[0], wav_file[timing[1]: timing[2]].raw_data])

        return fragments

    def __len__(self):
        return len(self.audio_fragments)

    def __getitem__(self, item: int):
        if self.transform:
            return self.transform(self.audio_fragments[item])
        return self.audio_fragments[item]


data = TIMITDataset(root_dir=r'E:\voice_datasets\timit\TIMIT_2\data', phone_codes=['ae'])
print(len(data))
print(data[0])
