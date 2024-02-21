import os
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# Load audio
def extract_wav(wav_path: str) -> np.ndarray:
    """
    The code above implements SAD, a pre-emphasis filter with a coefficient of 0.97, and normalization.

    :param wav_path: Path to the audio file.
    :type wav_path: str
    :return: Audio samples.
    :rtype: np.ndarray
    """
    # load the audio file
    s, sr = librosa.load(wav_path, sr=16000, mono=True)
    # split the audio file into speech segments
    speech_indices = librosa.effects.split(s, top_db=30)
    s = np.concatenate([s[start:end] for start, end in speech_indices])
    # apply a pre-emphasis filter
    s = librosa.effects.preemphasis(s, coef=0.97)
    # normalize
    s /= np.max(np.abs(s))
    return s


def load_audio(audio_path: str, utts: np.ndarray, nj: int = 12) -> list[np.ndarray]:
    """
    Load audio files from a directory.

    :param audio_path: Path to the directory of audio files.
    :type audio_path: str
    :param utts: List of utterance names with .wav extension.
    :type utts: np.ndarray
    :param nj: Number of cores for multi-threading.
    :type nj: int
    :return: List of audio samples.
    :rtype: list[np.ndarray]
    """
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
    return wavs
