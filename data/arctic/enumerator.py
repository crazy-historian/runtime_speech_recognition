import pandas as pd
from pathlib import Path

speaker_info = {
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


def get_description_table(
        speaker_nickname: str,
        speaker_l1: str,
        gender: str,
        annotation_dir_path: str,
        audio_dir_path: str
) -> list:
    table_rows = list()
    annotation_dir_path = Path(annotation_dir_path)
    for annotation_file in annotation_dir_path.iterdir():
        audio_file = Path(audio_dir_path, f'{annotation_file.stem}.wav')
        table_rows.append([
            speaker_nickname,
            speaker_l1,
            gender,
            str(annotation_file),
            str(audio_file)
        ])

    return table_rows


if __name__ == "__main__":
    dataset_file_path = "/media/maxim/Programming/voice_datasets/arctic/l2arctic_release_v5.0/data"
    table = list()
    for speaker_dir in Path(dataset_file_path).iterdir():
        table.extend(get_description_table(
            speaker_nickname=speaker_dir.stem,
            speaker_l1=speaker_info[speaker_dir.stem][0],
            gender=speaker_info[speaker_dir.stem][1],
            annotation_dir_path=str(Path(speaker_dir, 'annotation')),
            audio_dir_path=str(Path(speaker_dir, 'wav'))
        ))
    else:
        df = pd.DataFrame(data=table, columns=[
            'speaker_nickname', 'speaker_l1', 'speaker_gender', 'annotation_file_path', 'wav_file_path'
        ])
        df.to_csv('arctic_description.csv')
