from jiwer import wer
from asrecognition import ASREngine
from neuspell import BertChecker

MODEL_PATH = 'facebook/wav2vec2-base-960h'

class SpeechEngine():
    
    def __init__(self) -> None:
        self._model = ASREngine("en", model_path=MODEL_PATH)
        self._corrector = BertChecker()
        self._corrector.from_pretrained()
        
        
    def transcribe(self, audio_file_path:str, enable_correction=False)->str:
        transcription = self._model.transcribe([audio_file_path])[0]['transcription']
        if enable_correction:
            transcription = self.auto_correct(transcription)
        return transcription
    
    def auto_correct(self, text)->str:
        text = self._corrector.correct(text)
        return text