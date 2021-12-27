import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
import soundfile as sf

# get the path of assets (audio-files and transcripts)
ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
AUDIO_FILES_PATH = os.path.join(ASSETS_PATH, 'audio-files')
TRANSCRIPTS_PATH = os.path.join(ASSETS_PATH, 'originalText.xlsx')
LIBRISPEECH_DATASET_PATH = 'patrickvonplaten/librispeech_asr_dummy'


class DataLoader():
    
    def __init__(self, use_local=True) -> None:
        if use_local:
            self._audio_files = sorted(Path(AUDIO_FILES_PATH).glob('*.wav'), key=lambda x:x.name)
            self._transcripts = pd.read_excel(TRANSCRIPTS_PATH)['Reference'].values.tolist()
            if len(self._audio_files) != len(self._transcripts):
                raise Exception(f'Found {len(self._audio_files)} audio files and {len(self._transcripts)} transcription, they dont match')
        else:
            ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            ds = ds.map(self.__map_to_array)
            self._audio_files = [item['file'] for item in ds]
            self._transcripts = [item['text'] for item in ds]

    
    def __map_to_array(self, batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch


    def __len__(self):
        return len(self._audio_files)
        
    def get_data_by_id(self, id:int):
        '''
        accept an int id and return the audio and the transcript for it
        '''
        if (id < 0) or (id > len(self._audio_files)):
            return

        return str(self._audio_files[id]), self._transcripts[id]