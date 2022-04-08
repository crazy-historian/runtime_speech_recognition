import pandas as pd
import textgrid

from pathlib import Path
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Callable


class ArcticDataset(Dataset):
    """ ARCTIC L2 dataset """

    def __init__(self,
                 root_dir: str,
                 transform: Callable = None,
                 phone_codes: Union[List[str], str] = None,
                 gender: Optional[str] = None,
                 dialect: Optional[List[str]] = None):
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
        self.description_table = self._prepare_description()
        self.description_table = self._filter_description_table(gender, dialect)
        self.audio_fragments = self._get_audio_fragments()

    def _prepare_description(self) -> pd.DataFrame:
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

        df = pd.DataFrame(data=table, columns=[
            'nickname', 'l1', 'gender', 'labels_file_path', 'wav_file_path'
        ])
        df.to_csv('arctic_description.csv', index=False)

        return df

    def _filter_description_table(self, gender: Optional[str], dialect: Optional[str]) -> pd.DataFrame:
        if gender is not None:
            self.description_table = self.description_table.loc[
                self.description_table['gender'] == gender
            ]
        if dialect is not None:
            dialects = self.description_table['l1'].isin(dialect)
            self.description_table = self.description_table[dialects]

        return self.description_table

    def _get_audio_fragments(self) -> list:
        fragments = list()
        try:
            for _, file in self.description_table.iterrows():
                labels = textgrid.TextGrid.fromFile(file['labels_file_path'])
                timings = list()
                for interval in labels[1]:
                    if self.phone_codes is None or interval.mark in self.phone_codes:
                        start = interval.minTime * 1000
                        end = interval.maxTime * 1000
                        timings.append((interval.mark, start, end))
                if len(timings) > 0:
                    wav_file = AudioSegment.from_wav(file['wav_file_path'])
                    for timing in timings:
                        fragments.append([ timing[0], wav_file[timing[1]:timing[2]].raw_data])
        except ValueError:
            ...
        return fragments

    def __len__(self):
        return len(self.audio_fragments)

    def __getitem__(self, item: int):
        if self.transform:
            return self.transform(self.audio_fragments[item])
        return self.audio_fragments[item]

