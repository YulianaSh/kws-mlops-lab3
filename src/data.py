"""Завантаження датасету Google Speech Commands та перетворення аудіо у Mel-спектрограми."""
import hashlib
import io
import os
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

DEFAULT_URL = os.getenv(
    "DATA_URL",
    "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip")
DEFAULT_CLASSES = ["yes", "no", "up", "down"]

# Параметри препроцесингу (ті самі, що в ЛР №1)
SAMPLE_RATE = 16000
N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS = 512, 400, 160, 40
VAL_PERCENT, TEST_PERCENT = 10, 10

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH, n_mels=N_MELS)
_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)


# ---------------------------------------------------------------- датасет
def download_dataset(data_dir: Path, classes=DEFAULT_CLASSES, url: str = DEFAULT_URL) -> Path:
    """Завантажує mini_speech_commands.zip і розпаковує лише потрібні класи."""
    data_dir = Path(data_dir)
    if all((data_dir / c).is_dir() and any((data_dir / c).glob("*.wav")) for c in classes):
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Завантаження датасету: {url}", flush=True)
    with urllib.request.urlopen(url) as r:
        payload = r.read()
    print(f"Отримано {len(payload) / 2**20:.1f} МБ, розпаковка класів {classes}", flush=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) == 3 and parts[1] in classes and parts[2].endswith(".wav"):
                dst = data_dir / parts[1] / parts[2]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(name))
    return data_dir


def which_set(filename: str) -> str:
    """Офіційне розбиття Google: за хешем ID мовця (один мовець — в одній вибірці)."""
    speaker = re.sub(r"_nohash_.*$", "", filename)
    h = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 100
    if h < VAL_PERCENT:
        return "val"
    if h < VAL_PERCENT + TEST_PERCENT:
        return "test"
    return "train"


def list_files(data_dir: Path, classes, split: str):
    items = []
    for label, cls in enumerate(classes):
        for wav in sorted((Path(data_dir) / cls).glob("*.wav")):
            if which_set(wav.name) == split:
                items.append((wav, label))
    return items


# ---------------------------------------------------------------- аудіо
def load_wav(source) -> torch.Tensor:
    """WAV (шлях або байти) -> моно-сигнал рівно 1 с (16000 відліків)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    return torch.from_numpy(np.ascontiguousarray(audio[:SAMPLE_RATE]))


def to_logmel(wave: torch.Tensor) -> torch.Tensor:
    """[16000] -> [1, 40, 101] нормалізована лог-Mel-спектрограма."""
    spec = _to_db(_mel(wave))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.unsqueeze(0)


def wav_to_features(source) -> torch.Tensor:
    return to_logmel(load_wav(source))


class SpeechCommands(torch.utils.data.Dataset):
    def __init__(self, data_dir, classes, split, augment=False):
        self.items = list_files(data_dir, classes, split)
        self.augment = augment
        self.cache = [wav_to_features(p) for p, _ in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.cache[i]
        if self.augment:  # випадковий зсув у часі ±100 мс
            x = torch.roll(x, shifts=int(torch.randint(-10, 11, (1,))), dims=-1)
        return x, self.items[i][1]
