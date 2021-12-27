from tqdm.auto import tqdm
from jiwer import wer
from speech_engine import DataLoader, SpeechEngine
import numpy as np 
import matplotlib.pyplot as plt

data_loader = DataLoader(use_local=True)
engine = SpeechEngine()

raw_wer = []
corrected_wer = []
data_size = []

for i in tqdm(range(len(data_loader))):
    
    print(f'testing id {i}')
    audio_i, text_i = data_loader.get_data_by_id(i)
    text_raw = engine.transcribe(audio_i, enable_correction=False)
    text_corrected = engine.auto_correct(text_raw)
    raw_wer.append(wer(text_i, text_raw))
    corrected_wer.append(wer(text_i, text_corrected))
    data_size.append(i)


avg_raw_wer = sum(raw_wer) / len(raw_wer)
avg_corr_wer = sum(corrected_wer) / len(corrected_wer)
print(f'avg raw wer : {avg_raw_wer:.2f}')
print(f'avg corr wer: {avg_corr_wer:.2f}')
print('--'*10)
print('raw wers :', ', '.join([str(round(i, 2)) for i in raw_wer]))
print('--'*10)
print('corr wers:', ', '.join([str(round(i, 2)) for i in corrected_wer]))

# Visualize the data
X_axis = np.arange(len(data_loader))

plt.bar(X_axis - 0.2, raw_wer, 0.4, label = 'asr_wer')
plt.bar(X_axis + 0.2, corrected_wer, 0.4, label = 'corr_wer')

plt.xticks(X_axis, data_size)
plt.xlabel("Audio files")
plt.ylabel("WER")
plt.title("WER after speech and correction models")
plt.legend()
plt.savefig('figures/audio-files_wer_base_bert.png')
plt.show()
