import pandas as pd
import textgrid

from pathlib import Path
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Callable


class WavToBytes:
    """
        Read WAV file and return raw audio data in bytes.

    """

    def __init__(self, phone_codes: Union[List[str], str] = None):
        self.phone_codes = phone_codes

    def __call__(self, file_name: str, textgrid_labels_file: str):
        wav_file = AudioSegment.from_wav(file_name)
        if self.phone_codes:
            wav_fragments = dict()
            labels = textgrid.TextGrid.fromFile(textgrid_labels_file)
            for interval in labels[1]:
                if interval.mark not in self.phone_codes:
                    continue
                start = interval.minTime * 1000
                end = interval.maxTime * 1000
                if interval.mark not in wav_fragments:
                    wav_fragments[interval.mark] = list()
                wav_fragments[interval.mark].append(wav_file[start:end].raw_data)
            return wav_fragments
        else:
            return wav_file[:].raw_data


class ArcticDataset(Dataset):
    """ ARCTIC L2 dataset """

    def __init__(self,
                 root_dir: str,
                 transform: Callable = WavToBytes(),
                 gender: Optional[Union[str, List[str]]] = None,
                 dialect: Optional[Union[str, List[str]]] = None):
        self.root_dir = root_dir
        self.transform = transform
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
        self.gender = gender
        self.dialect = dialect

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

        return pd.DataFrame(data=table, columns=[
            'speaker_nickname', 'speaker_l1', 'speaker_gender', 'annotation_file_path', 'wav_file_path'
        ])

    def __len__(self):
        return self.description_table.shape[1]

    def __getitem__(self, item: int):
        return self.transform(
            file_name=self.description_table.iloc[item]['wav_file_path'],
            textgrid_labels_file=self.description_table.iloc[item]['annotation_file_path']
        )




dataset = ArcticDataset(r'E:\voice_datasets\arctic\l2arctic_release_v5.0\data')
print(dataset[0])
