import pandas as pd
from typing import Optional, Union, List, Dict
from abc import abstractmethod, ABC


def merge_dictionaries(first: dict, second: dict) -> dict:
    for key, value in second.items():
        if key in first:
            first[key].extend(second[key])
        else:
            first[key] = second[key]
    return first


class VoiceDataDict(ABC):
    def __init__(self):
        self.phoneme = None
        self.description_table = None

    @abstractmethod
    def prepare_description_table(self, dataset_path: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def cut_phonemes_from_file(self, *args, **kwargs) -> Dict[str, List[bytes]]:
        ...

    @abstractmethod
    def get_phonemes(self, *phonemes_codes: Union[List[str], str]) -> dict:
        ...

