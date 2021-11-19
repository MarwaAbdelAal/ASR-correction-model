from jiwer import wer
from speech_engine import DataLoader, SpeechEngine

data_loader = DataLoader()
engine = SpeechEngine()

raw_wer = []
corrected_wer = []

for i in range(len(data_loader)):
    
    print(f'testing id {i}')
    audio_i, text_i = data_loader.get_data_by_id(i)
    text_raw = engine.transcribe(audio_i, enable_correction=False)
    text_corrected = engine.auto_correct(text_raw)
    raw_wer.append(wer(text_i, text_raw))
    corrected_wer.append(wer(text_i, text_corrected))


avg_raw_wer = sum(raw_wer)/len(raw_wer)
avg_corr_wer = sum(corrected_wer)/len(corrected_wer)
print(f'avg raw wer : {avg_raw_wer:.2f}')
print(f'avg corr wer: {avg_corr_wer:.2f}')
print('--'*10)
print('raw wers :', ','.join([str(round(i, 2)) for i in raw_wer]))
print('corr wers:', ','.join([str(round(i, 2)) for i in corrected_wer]))