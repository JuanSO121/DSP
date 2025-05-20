import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import soundfile as sf

def matlab_style_filter_design(fs=44100):
    """
    Implementa un filtro pasa-banda al estilo MATLAB usando el método de 
    muestreo en frecuencia con la función sinc
    """
    # Parámetros del filtro
    M = 66  # Orden del filtro (como en tu ejemplo MATLAB)
    n = np.arange(-M/2, M/2+1)  # Vector de muestras
    
    # Frecuencias de corte normalizadas (comparable a tu ejemplo MATLAB)
    wc1 = 0.24 * np.pi  # Frecuencia de corte superior (paso bajo 1)
    wc2 = 0.6 * np.pi   # Frecuencia de corte inferior (paso bajo 2)
    
    # h(n) del filtro pasa-bajas con wc1
    hn1 = (wc1/np.pi) * np.sinc(wc1*n/np.pi)
    
    # h(n) del filtro pasa-bajas con wc2
    hn2 = (wc2/np.pi) * np.sinc(wc2*n/np.pi)
    
    # h(n) del filtro pasa-banda (resta de los dos filtros pasa-bajas)
    hn = hn2 - hn1
    
    # Aplicamos ventana de Hanning
    win = np.hanning(len(n))
    hn_windowed = hn * win
    
    # Visualizar la respuesta al impulso
    plt.figure(figsize=(10, 6))
    plt.stem(n, hn_windowed)
    plt.title('Respuesta al impulso truncada x ventana')
    plt.xlabel('n [muestra]')
    plt.grid(True)
    plt.show()
    
    # Respuesta en Frecuencia H(w)
    N1 = 10000
    H1jw = np.fft.fftshift(np.abs(np.fft.fft(hn_windowed, N1)))
    w = np.arange(-N1/2, N1/2) * (2*np.pi/N1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(w/np.pi, H1jw)  # Escala Lineal
    plt.grid(True)
    plt.title('Respuesta en frecuencia')
    plt.xlabel('w/pi [rad/s]')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(w/np.pi, 20*np.log10(H1jw + 1e-10))  # Escala en dB, añadimos 1e-10 para evitar log(0)
    plt.grid(True)
    plt.title('Respuesta en frecuencia en dB')
    plt.xlabel('w/pi [rad/s]')
    plt.show()
    
    return hn_windowed

def apply_filter_to_audio(audio_file, filter_coeffs, fs=44100):
    """
    Aplica el filtro diseñado a un archivo de audio
    """
    # Cargar archivo de audio
    try:
        data, sample_rate = sf.read(audio_file)
        if len(data.shape) > 1:
            # Convertir estéreo a mono si es necesario
            data = np.mean(data, axis=1)
        
        # Aplicar el filtro utilizando convolución
        filtered_audio = signal.lfilter(filter_coeffs, 1, data)
        
        # Normalizar la amplitud para evitar distorsión
        max_amplitude = np.max(np.abs(filtered_audio))
        if max_amplitude > 0:
            filtered_audio = filtered_audio / max_amplitude * 0.9
        
        # Guardar audio filtrado
        output_file = 'audio_filtrado_matlab_style.wav'
        sf.write(output_file, filtered_audio, sample_rate)
        
        print(f"Audio filtrado guardado como '{output_file}'")
        
        # Visualizar espectrograma del audio original y filtrado
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.specgram(data, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
        plt.title('Espectrograma del Audio Original')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.colorbar(label='Intensidad (dB)')
        
        plt.subplot(2, 1, 2)
        plt.specgram(filtered_audio, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
        plt.title('Espectrograma del Audio Filtrado')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.colorbar(label='Intensidad (dB)')
        
        plt.tight_layout()
        plt.show()
        
        # ¿Reproducir audio?
        play_audio = input("¿Quieres reproducir el audio filtrado? (s/n): ")
        if play_audio.lower() == 's':
            print("Reproduciendo audio filtrado...")
            sd.play(filtered_audio, sample_rate)
            sd.wait()  # Esperar a que termine la reproducción
        
        return filtered_audio
        
    except Exception as e:
        print(f"Error al procesar el archivo de audio: {e}")
        return None

def main():
    print("\n=== DISEÑO DE FILTRO PASA-BANDA AL ESTILO MATLAB EN PYTHON ===")
    
    # Diseñar el filtro
    print("Diseñando filtro pasa-banda...")
    filter_coeffs = matlab_style_filter_design()
    
    # Preguntar si se desea aplicar a un archivo de audio
    apply_to_audio = input("\n¿Quieres aplicar este filtro a un archivo de audio? (s/n): ")
    if apply_to_audio.lower() == 's':
        audio_file = input("Ingresa la ruta del archivo de audio (por defecto: audio_original.wav): ") or "audio_original.wav"
        apply_filter_to_audio(audio_file, filter_coeffs)
    
    # Ejemplo de señal de prueba (como en MATLAB)
    print("\nAplicando filtro a una señal senoidal de prueba...")
    n1 = np.arange(300)
    w0 = 0.2 * np.pi  # Frecuencia angular normalizada
    xn = np.cos(w0 * n1)
    
    # Filtrar usando convolución
    yn = np.convolve(xn, filter_coeffs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(yn, 'r')
    plt.title('Señal senoidal filtrada con conv')
    plt.xlabel('n [muestras]')
    plt.grid(True)
    plt.show()
    
    # Filtrar usando filter (como en MATLAB)
    yn1 = signal.lfilter(filter_coeffs, 1, xn)
    
    plt.figure(figsize=(10, 6))
    plt.plot(yn1, 'r')
    plt.title('Señal senoidal filtrada con filter')
    plt.xlabel('n [muestras]')
    plt.grid(True)
    plt.show()
    
    # Comparar FFT de señal original y filtrada
    N_fft = 1024
    X = np.fft.fftshift(np.abs(np.fft.fft(xn, N_fft)))
    Y = np.fft.fftshift(np.abs(np.fft.fft(yn1, N_fft)))
    freq = np.arange(-N_fft/2, N_fft/2) * (2*np.pi/N_fft)
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq/np.pi, X, 'b', label='Original')
    plt.plot(freq/np.pi, Y, 'r', label='Filtrada')
    plt.grid(True)
    plt.legend()
    plt.title('Espectro de frecuencia')
    plt.xlabel('w/pi [rad/s]')
    plt.show()

if __name__ == "__main__":
    main()