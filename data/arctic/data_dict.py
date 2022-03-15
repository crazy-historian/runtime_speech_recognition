import textgrid
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
from typing import Union, List, Dict

from data.voice_data import VoiceDataDict, merge_dictionaries


class ArcticVoiceData(VoiceDataDict):
    def __init__(self):
        super().__init__()
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

    def prepare_description_table(self, dataset_path: str) -> None:
        table = list()
        for speaker_dir in Path(dataset_path).iterdir():
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

        self.description_table = pd.DataFrame(data=table, columns=[
            'speaker_nickname', 'speaker_l1', 'speaker_gender', 'annotation_file_path', 'wav_file_path'
        ])

    def cut_phonemes_from_file(
            self,
            audio_file_path: str,
            annotation_file_path: str,
            arpabet_phone_code: Union[List[str], str] = None
    ) -> Dict[str, List[bytes]]:

        if isinstance(arpabet_phone_code, str):
            arpabet_phone_code = [arpabet_phone_code]

        tg_labels = textgrid.TextGrid.fromFile(annotation_file_path)
        wav_file = AudioSegment.from_wav(audio_file_path)
        wav_fragments = dict()

        for interval in tg_labels[1]:
            if arpabet_phone_code is not None and interval.mark not in arpabet_phone_code:
                continue
            start = interval.minTime * 1000
            end = interval.maxTime * 1000
            if interval.mark not in wav_fragments:
                wav_fragments[interval.mark] = list()

            wav_fragments[interval.mark].append(wav_file[start:end].raw_data)

        return wav_fragments

    def get_phonemes(self, *phoneme_codes: Union[List[str], str]):
        phonemes = dict()
        if self.description_table is not None:
            for index, row in self.description_table.iterrows():
                try:
                    raw = self.cut_phonemes_from_file(
                        audio_file_path=row['wav_file_path'],
                        annotation_file_path=row['annotation_file_path'],
                        arpabet_phone_code=['AH0', 'AH1', 'IH0', 'IH1']
                    )
                    phonemes = merge_dictionaries(phonemes, raw)
                except ValueError as err:
                    print(err)
        return phonemes
