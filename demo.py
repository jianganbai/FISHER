import torch
import torchaudio
import torch.nn.functional as F

from models.fisher import FISHER


wav, sr = torchaudio.load('quich_infer/demo.wav')
# You can replace it with your custom loading function for other signals

wav = wav - wav.mean()
STFT = torchaudio.transforms.Spectrogram(
    n_fft=25 * sr // 1000,
    win_length=None,
    hop_length=10 * sr // 1000,
    power=1,
    center=False
)
spec = torch.log(torch.abs(STFT(wav)) + 1e-10)
spec = spec.transpose(-2, -1)  # [1, time, freq]
spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)

model_path = '/path/to/local/fisher/model.pt'
model = FISHER.from_pretrained(model_path)
model = model.cuda()
model.eval()

# time-wise padding
if spec.shape[-2] < 1024:
    spec = F.pad(spec, (0, 0, 0, 1024 - spec.shape[-2]))
else:
    spec = spec[:, :1024]
# freq-wise padding
if spec.shape[-1] < model.cfg.band_width:
    spec = F.pad(spec, (0, model.cfg.band_width - spec.shape[-1]))
spec = spec.unsqueeze(1).cuda()

with torch.no_grad():
    # Use autocast for mixed precision inference. You can disable it for full precision.
    with torch.autocast('cuda'):
        repre = model.extract_features(spec)
print(repre.shape)
