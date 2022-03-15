import textgrid
from pydub import AudioSegment
from typing import Union, List, Dict


def get_phone_fragments(
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
