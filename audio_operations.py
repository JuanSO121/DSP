import numpy as np
from scipy import signal
import soundfile as sf

def design_lpf_fir(fs, cutoff=3000, numtaps=101):
    """Diseña un filtro paso-bajo FIR"""
    nyq = fs / 2
    return signal.firwin(numtaps, cutoff / nyq, window="hamming")

def design_hpf_fir(fs, cutoff=300, numtaps=101):
    """Diseña un filtro paso-alto FIR"""
    nyq = fs / 2
    return signal.firwin(numtaps, cutoff / nyq, pass_zero=False, window="hamming")

def design_peaking_iir(fs, f0, gain_db, Q=1):
    """Diseña un filtro IIR peak/notch"""
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

class EQProcessor:
    """Clase para procesar audio con ecualizador en tiempo real"""
    def __init__(self, fs, buffer_size=1024):
        self.fs = fs
        self.buffer_size = buffer_size
        
        # Estados iniciales para filtros
        self.lpf_coefficients = None
        self.hpf_coefficients = None
        self.band_filters = []
        
        # Estados para los filtros
        self.lpf_zi = None
        self.hpf_zi = None
        self.band_zis = []
        
        # Configuración por defecto
        self.configure({
            "lpf_cutoff": 4000,
            "hpf_cutoff": 200,
            "bands": [
                {"f0": 1000, "gain": 0, "Q": 1.0},
                {"f0": 3000, "gain": 0, "Q": 1.0},
                {"f0": 5000, "gain": 0, "Q": 1.0}
            ]
        })
        
    def configure(self, eq_settings):
        """Configura los filtros según los ajustes de EQ"""
        # Reiniciar los estados si cambian los filtros
        reset_states = (self.lpf_coefficients is None)
        
        # Configurar LPF
        self.lpf_coefficients = design_lpf_fir(self.fs, cutoff=eq_settings["lpf_cutoff"], numtaps=65)
        
        # Configurar HPF
        self.hpf_coefficients = design_hpf_fir(self.fs, cutoff=eq_settings["hpf_cutoff"], numtaps=65)
        
        # Configurar filtros IIR de bandas
        self.band_filters = []
        for band in eq_settings["bands"]:
            b, a = design_peaking_iir(self.fs, band["f0"], band["gain"], band["Q"])
            self.band_filters.append((b, a))
        
        # Inicializar o reiniciar estados
        if reset_states or len(self.lpf_zi) != len(self.lpf_coefficients) - 1:
            self.reset_states()
            
    def reset_states(self):
        """Reinicia los estados de los filtros"""
        # Inicializar estados para filtros FIR
        self.lpf_zi = np.zeros(len(self.lpf_coefficients) - 1)
        self.hpf_zi = np.zeros(len(self.hpf_coefficients) - 1)
        
        # Inicializar estados para filtros IIR
        self.band_zis = []
        for b, a in self.band_filters:
            zi = signal.lfilter_zi(b, a)
            self.band_zis.append(zi)
    
    def process_block(self, audio_block):
        """Procesa un bloque de audio, manteniendo estados entre bloques"""
        if audio_block is None or len(audio_block) == 0:
            return np.zeros(self.buffer_size)
            
        # Normalización suave del bloque de entrada
        # Evitamos normalizar al máximo para prevenir fluctuaciones de volumen
        max_amp = np.max(np.abs(audio_block))
        if max_amp > 0.95:  # Solo normalizamos si está a punto de recortar
            audio_block = audio_block * (0.9 / max_amp)
        
        # Aplicar LPF
        filtered, self.lpf_zi = signal.lfilter(
            self.lpf_coefficients, 1, audio_block, zi=self.lpf_zi
        )
        
        # Aplicar HPF
        filtered, self.hpf_zi = signal.lfilter(
            self.hpf_coefficients, 1, filtered, zi=self.hpf_zi
        )
        
        # Aplicar filtros IIR
        for i, (b, a) in enumerate(self.band_filters):
            filtered, self.band_zis[i] = signal.lfilter(
                b, a, filtered, zi=self.band_zis[i]
            )
        
        # Aplicar suave compresión dinámica para evitar recortes
        filtered = apply_soft_knee_compression(filtered, threshold=0.7, ratio=4.0)
            
        return filtered

def apply_equalizer(audio, fs, eq_settings):
    """Aplica el ecualizador a un archivo completo (no en tiempo real)"""
    if audio is None:
        return None
        
    # Crear procesador
    processor = EQProcessor(fs)
    processor.configure(eq_settings)
    
    # Procesar por bloques para simular tiempo real
    block_size = 1024
    result = np.zeros_like(audio)
    
    for i in range(0, len(audio), block_size):
        end = min(i + block_size, len(audio))
        result[i:end] = processor.process_block(audio[i:end])
    
    return result

def apply_soft_knee_compression(audio, threshold=0.5, ratio=3.0, knee_width=0.1):
    """Aplica compresión con curva suave para un procesamiento más natural"""
    output = np.zeros_like(audio)
    
    for i in range(len(audio)):
        x = np.abs(audio[i])
        if x < threshold - knee_width/2:
            # Por debajo del umbral
            gain = 1.0
        elif x > threshold + knee_width/2:
            # Por encima del umbral
            gain = 1.0 + (1.0/ratio - 1.0) * (x - threshold)
        else:
            # En la zona de la rodilla
            knee_factor = (x - (threshold - knee_width/2)) / knee_width
            gain = 1.0 + (1.0/ratio - 1.0) * knee_factor**2 * knee_width/2
            
        # Aplicar ganancia pero preservar el signo
        output[i] = audio[i] / (1.0 if gain == 0 else gain)
    
    return output

def apply_compressor(audio, threshold_db=-20, ratio=4.0, attack_ms=5, release_ms=50):
    """Aplica un compresor de dinámica al audio"""
    if audio is None:
        return None

    # Convertir parámetros a escala lineal
    threshold = 10 ** (threshold_db / 20)
    attack = 1 - np.exp(-1.0 / (attack_ms * fs / 1000.0))
    release = 1 - np.exp(-1.0 / (release_ms * fs / 1000.0))
    
    # Inicializar envelope
    envelope = 0
    output = np.zeros_like(audio)
    
    # Procesar muestra por muestra
    for i in range(len(audio)):
        # Detector (rectificador de onda completa)
        abs_sample = np.abs(audio[i])
        
        # Generador de envelope (detector de nivel)
        if abs_sample > envelope:
            envelope = attack * abs_sample + (1 - attack) * envelope
        else:
            envelope = release * abs_sample + (1 - release) * envelope
            
        # VCA (procesador de ganancia)
        if envelope > threshold:
            # Calcular ganancia de compresión
            gain_reduction = threshold + (envelope - threshold) / ratio
            gain = gain_reduction / envelope
        else:
            gain = 1.0
            
        # Aplicar ganancia
        output[i] = audio[i] * gain
    
    # Normalizar la salida
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp * 0.9
        
    return output