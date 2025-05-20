import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from audio_operations import design_lpf_fir, design_hpf_fir, design_peaking_iir
from scipy import signal


def visualize_time(audio, fs, title="Audio en el Tiempo"):
    plt.figure(figsize=(10, 3))
    time_axis = np.arange(0, len(audio)) / fs
    plt.plot(time_axis, audio)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()

def visualize_frequency(audio, fs, title="Espectro de Frecuencia"):
    N = len(audio)
    X = fft(audio)
    X_mag = np.abs(X[:N//2]) / N
    freq = np.linspace(0, fs/2, N//2)
    plt.figure(figsize=(10, 3))
    plt.semilogx(freq, 20 * np.log10(X_mag + 1e-10))
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(True)
    plt.show()

def visualize_spectrogram(audio, fs, title="Espectrograma"):
    plt.figure(figsize=(10, 5))
    plt.specgram(audio, NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.colorbar(label="Intensidad (dB)")
    plt.ylim([0, 8000])
    plt.show()


def visualize_eq_response(fs, eq_settings):
    """Muestra la respuesta en frecuencia del ecualizador completo"""
    w = np.linspace(0, np.pi, 2048)

    # Inicializamos la respuesta total
    w_total, h_total = w, np.ones_like(w, dtype=complex)

    # LPF
    lpf = design_lpf_fir(fs, eq_settings["lpf_cutoff"])
    _, h_lpf = signal.freqz(lpf, worN=w)
    h_total *= h_lpf

    # HPF
    hpf = design_hpf_fir(fs, eq_settings["hpf_cutoff"])
    _, h_hpf = signal.freqz(hpf, worN=w)
    h_total *= h_hpf

    # IIR bands
    for band in eq_settings["bands"]:
        b, a = design_peaking_iir(fs, band["f0"], band["gain"], band["Q"])
        _, h = signal.freqz(b, a, worN=w)
        h_total *= h

    # Convertimos frecuencia a Hz
    freqs = w * fs / (2 * np.pi)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, 20 * np.log10(np.abs(h_total) + 1e-6))
    plt.title("Curva del Ecualizador (respuesta en frecuencia)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Ganancia (dB)")
    plt.grid(True)
    plt.ylim([-20, 20])
    plt.xlim([20, fs / 2])
    plt.tight_layout()
    plt.show()

