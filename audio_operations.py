import numpy as np
from scipy import signal
from scipy.fft import fft
import soundfile as sf



def design_bandpass_filter(fs, lowcut=300, highcut=3400, order=101):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b = signal.firwin(order, [low, high], pass_zero=False, window='hamming')
    return b

def apply_filter(audio, b):
    filtered = signal.lfilter(b, 1, audio)
    max_amp = np.max(np.abs(filtered))
    if max_amp > 0:
        filtered = filtered / max_amp * 0.9
    sf.write('audio_filtrado.wav', filtered, 44100)
    return filtered

def apply_noise_reduction(audio, fs, noise_level=0.5):
    noise_sample_size = int(0.1 * fs)
    noise_sample = audio[:noise_sample_size]
    noise_fft = fft(noise_sample)
    noise_power = np.abs(noise_fft) ** 2
    signal_fft = fft(audio)
    signal_power = np.abs(signal_fft) ** 2
    reduction = np.maximum(1 - noise_level * noise_power.mean() / (signal_power + 1e-10), 0)
    filtered_fft = signal_fft * reduction
    filtered = np.real(np.fft.ifft(filtered_fft))
    max_amp = np.max(np.abs(filtered))
    if max_amp > 0:
        filtered = filtered / max_amp * 0.9
    sf.write('audio_sin_ruido.wav', filtered, fs)
    return filtered


def design_lpf_fir(fs, cutoff=3000, numtaps=101):
    nyq = fs / 2
    return signal.firwin(numtaps, cutoff / nyq, window="hamming")

def design_hpf_fir(fs, cutoff=300, numtaps=101):
    nyq = fs / 2
    return signal.firwin(numtaps, cutoff / nyq, pass_zero=False, window="hamming")

def design_peaking_iir(fs, f0, gain_db, Q=1):
    A = 10**(gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    return b, a


def apply_equalizer(audio, fs, eq_settings):
    """
    Aplica 5 filtros en secuencia: 2 FIR (LPF, HPF) y 3 IIR peaking.
    eq_settings = {
        "lpf_cutoff": ...,
        "hpf_cutoff": ...,
        "bands": [
            {"f0": ..., "gain": ..., "Q": ...},
            ...
        ]
    }
    """
    if audio is None:
        return None

    # LPF
    lpf = design_lpf_fir(fs, cutoff=eq_settings["lpf_cutoff"])
    filtered = signal.lfilter(lpf, 1, audio)

    # HPF
    hpf = design_hpf_fir(fs, cutoff=eq_settings["hpf_cutoff"])
    filtered = signal.lfilter(hpf, 1, filtered)

    # 3 peaking filters
    for band in eq_settings["bands"]:
        b, a = design_peaking_iir(fs, band["f0"], band["gain"], band["Q"])
        filtered = signal.lfilter(b, a, filtered)

    # Normalizamos
    max_amp = np.max(np.abs(filtered))
    if max_amp > 0:
        filtered = filtered / max_amp * 0.9

    return filtered

