import os
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np

from whisper.utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 10.24
N_SAMPLES = int(CHUNK_LENGTH * SAMPLE_RATE)  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array: np.ndarray, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int = N_MELS) -> np.ndarray:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return f[f"mel_{n_mels}"]

def sliding_window_view(x, window_shape, step=1):
    shape = ((x.shape[-1] - window_shape) // step + 1,) + (window_shape,)
    strides = (step * x.strides[-1],) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def numpy_stft(audio: np.ndarray, N_FFT: int, HOP_LENGTH: int):
    window = np.hanning(N_FFT)
    num_frames = 1 + (audio.size - N_FFT) // HOP_LENGTH
    if (audio.size - N_FFT) % HOP_LENGTH > 0:
        num_frames += 1
    audio_padded = np.pad(audio, pad_width=(N_FFT//2, N_FFT//2), mode='constant')
    frames = sliding_window_view(audio_padded, N_FFT, HOP_LENGTH)
    frames = frames[:num_frames]
    stft = np.fft.rfft(frames * window, axis=-1)

    cpstft = (np.abs(stft[:,:N_FFT//2 + 1]) ** 2).T
    magnitudes = cpstft.astype(audio.dtype)
    return magnitudes


def log_mel_spectrogram(audio: Union[str, np.ndarray], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray], shape = (*)
        The path to audio or either a NumPy array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    np.ndarray, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)

    magnitudes = numpy_stft(audio, N_FFT, HOP_LENGTH)

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
