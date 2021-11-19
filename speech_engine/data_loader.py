import os
from pathlib import Path
import pandas as pd

ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
AUDIO_FILES_PATH = os.path.join(ASSETS_PATH, 'audio-files')
TRANSCRIPTS_PATH = os.path.join(ASSETS_PATH, 'originalText.xlsx')


class DataLoader():
    
    def __init__(self) -> None:
        self._audio_files = sorted(Path(AUDIO_FILES_PATH).glob('*.wav'), key=lambda x:x.name)
        self._transcripts = pd.read_excel(TRANSCRIPTS_PATH)['Reference'].values.tolist()
        if len(self._audio_files) != len(self._transcripts):
            raise Exception(f'Found {len(self._audio_files)} audio files and {len(self._transcripts)} transcription, they dont match')
        
    def __len__(self):
        return len(self._audio_files)
        
    def get_data_by_id(self, id:int):
        '''
        accept an int id and return the audio and the transcript for it
        '''
        if id < 0 or id > len(self._audio_files):
            return ()

        return str(self._audio_files[id]), self._transcripts[id]