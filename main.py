import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import soundfile as sf
import time
from scipy.fft import fft, fftshift
import os

class AudioProcessor:
    def __init__(self):
        # Parámetros de grabación
        self.fs = 44100  # Frecuencia de muestreo en Hz
        self.duration = 5  # Duración de la grabación en segundos
        self.audio_data = None
        self.filtered_audio = None
        
        # Verificar que las bibliotecas necesarias estén instaladas
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            print("\n¡ATENCIÓN! Las bibliotecas necesarias no están instaladas.")
            print("Para instalar las dependencias, ejecuta:")
            print("pip install numpy scipy matplotlib sounddevice soundfile")
            print("Si sigues teniendo problemas, consulta la documentación de cada biblioteca.\n")
        
    def record_audio(self):
        """Grabar audio del micrófono"""
        print(f"Grabando audio por {self.duration} segundos...")
        self.audio_data = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()  # Esperar hasta que termine la grabación
        self.audio_data = self.audio_data.flatten()  # Convertir a array 1D
        print("Grabación completada.")
        
        # Guardar el audio original
        sf.write('audio_original.wav', self.audio_data, self.fs)
        print("Audio original guardado como 'audio_original.wav'")
        
        return self.audio_data
    
    def load_audio(self, file_path):
        """Cargar audio desde un archivo"""
        try:
            # Cargar el archivo y obtener tanto los datos como la frecuencia de muestreo
            self.audio_data, new_fs = sf.read(file_path, always_2d=False)
            
            # Actualizar la frecuencia de muestreo según el archivo cargado
            self.fs = new_fs
            
            print(f"Audio cargado: {file_path}")
            print(f"Frecuencia de muestreo: {self.fs} Hz")
            print(f"Duración: {len(self.audio_data) / self.fs:.2f} segundos")
            
            # Si el audio es estéreo (2 canales), convertir a mono
            if len(self.audio_data.shape) > 1 and self.audio_data.shape[1] > 1:
                print("Convirtiendo audio estéreo a mono...")
                self.audio_data = np.mean(self.audio_data, axis=1)
                
            return self.audio_data
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None
    
    def play_audio(self, audio=None):
        """Reproducir audio"""
        if audio is None:
            audio = self.audio_data
            
        if audio is not None:
            print(f"Reproduciendo audio con frecuencia de muestreo: {self.fs} Hz")
            sd.play(audio, self.fs)
            status = sd.wait()  # Esperar a que termine la reproducción y capturar estado
            if status:
                print(f"Advertencia durante la reproducción: {status}")
        else:
            print("No hay audio para reproducir")
    
    def visualize_time_domain(self, audio=None, title="Señal de Audio en Dominio del Tiempo"):
        """Visualizar la señal en el dominio del tiempo"""
        if audio is None:
            audio = self.audio_data
            
        if audio is not None:
            plt.figure(figsize=(12, 4))
            time_axis = np.arange(0, len(audio)) / self.fs
            plt.plot(time_axis, audio)
            plt.title(title)
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.grid(True)
            plt.show()
        else:
            print("No hay audio para visualizar")
    
    def visualize_frequency_domain(self, audio=None, title="Espectro de Frecuencia"):
        """Visualizar el espectro de frecuencia de la señal"""
        if audio is None:
            audio = self.audio_data
            
        if audio is not None:
            N = len(audio)
            X = fft(audio)
            X_mag = np.abs(X[:N//2]) / N  # Normalizamos y tomamos solo las frecuencias positivas
            freq = np.linspace(0, self.fs/2, N//2)
            
            plt.figure(figsize=(12, 4))
            plt.semilogx(freq, 20 * np.log10(X_mag + 1e-10))  # Convertimos a dB
            plt.title(title)
            plt.xlabel('Frecuencia (Hz)')
            plt.ylabel('Magnitud (dB)')
            plt.grid(True)
            plt.xlim([20, self.fs/2])  # Limitamos a rango audible
            plt.show()
        else:
            print("No hay audio para visualizar")
    
    def create_spectrogram(self, audio=None, title="Espectrograma"):
        """Crear espectrograma de la señal"""
        if audio is None:
            audio = self.audio_data
            
        if audio is not None:
            plt.figure(figsize=(12, 6))
            plt.specgram(audio, NFFT=1024, Fs=self.fs, noverlap=512, cmap='viridis')
            plt.title(title)
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Frecuencia (Hz)')
            plt.colorbar(label='Intensidad (dB)')
            plt.ylim([0, 8000])  # Limitamos a rango de voz humana
            plt.show()
        else:
            print("No hay audio para visualizar")
    
    def design_bandpass_filter(self, lowcut=300, highcut=3400):
        """Diseñar un filtro pasa-banda para la voz humana"""
        # Frecuencias normalizadas
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Orden del filtro
        order = 6
        
        # Diseño del filtro FIR usando ventana
        numtaps = 101
        b = signal.firwin(numtaps, [low, high], pass_zero=False, window='hamming')
        
        # Visualizar la respuesta en frecuencia del filtro
        w, h = signal.freqz(b, 1, worN=8000)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * self.fs * w / np.pi, np.abs(h))
        plt.title('Respuesta en Frecuencia del Filtro Pasa-banda')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Ganancia')
        plt.grid(True)
        plt.axvline(lowcut, color='r', alpha=0.5)
        plt.axvline(highcut, color='r', alpha=0.5)
        
        plt.subplot(2, 1, 2)
        plt.plot(0.5 * self.fs * w / np.pi, 20 * np.log10(np.abs(h) + 1e-10))
        plt.title('Respuesta en Frecuencia (dB)')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Ganancia (dB)')
        plt.grid(True)
        plt.axvline(lowcut, color='r', alpha=0.5)
        plt.axvline(highcut, color='r', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        return b
    
    def apply_filter(self, filter_coeffs, audio=None):
        """Aplicar el filtro a la señal de audio"""
        if audio is None:
            audio = self.audio_data
            
        if audio is not None:
            # Aplicar el filtro
            self.filtered_audio = signal.lfilter(filter_coeffs, 1, audio)
            
            # Normalizar la señal filtrada para evitar distorsión
            max_amplitude = np.max(np.abs(self.filtered_audio))
            if max_amplitude > 0:
                self.filtered_audio = self.filtered_audio / max_amplitude * 0.9
            
            # Guardar el audio filtrado
            sf.write('audio_filtrado.wav', self.filtered_audio, self.fs)
            print("Audio filtrado guardado como 'audio_filtrado.wav'")
            
            return self.filtered_audio
        else:
            print("No hay audio para filtrar")
            return None
    
    def apply_noise_reduction(self, noise_level=0.1):
        """Aplicar reducción de ruido usando un filtro adaptativo"""
        if self.audio_data is not None:
            # Estimamos el espectro del ruido usando los primeros segundos del audio (suponiendo que hay silencio)
            noise_sample_duration = 0.1  # 0.5 segundos iniciales
            noise_sample_size = min(int(noise_sample_duration * self.fs), len(self.audio_data) // 4)
            noise_sample = self.audio_data[:noise_sample_size]  
            
            # Calculamos la FFT del ruido
            noise_fft = fft(noise_sample)
            noise_power = np.abs(noise_fft) ** 2
            
            # Calculamos la FFT de la señal completa
            signal_fft = fft(self.audio_data)
            signal_power = np.abs(signal_fft) ** 2
            
            # Factor de reducción de ruido adaptativo
            # Usamos el promedio del espectro de ruido y lo aplicamos a cada frecuencia
            noise_mean_power = noise_power.mean()
            reduction_factor = np.maximum(1 - noise_level * noise_mean_power / (signal_power + 1e-10), 0)
            
            # Aplicamos el factor de reducción
            filtered_fft = signal_fft * reduction_factor
            
            # Volvemos al dominio del tiempo
            self.filtered_audio = np.real(np.fft.ifft(filtered_fft))
            
            # Normalizar la señal filtrada para evitar distorsión
            max_amplitude = np.max(np.abs(self.filtered_audio))
            if max_amplitude > 0:
                self.filtered_audio = self.filtered_audio / max_amplitude * 0.9
            
            # Guardar el audio filtrado
            sf.write('audio_sin_ruido.wav', self.filtered_audio, self.fs)
            print("Audio con reducción de ruido guardado como 'audio_sin_ruido.wav'")
            
            return self.filtered_audio
        else:
            print("No hay audio para procesar")
            return None

def main():
    processor = AudioProcessor()
    
    print("\n=== PROCESADOR DE AUDIO PARA ELIMINAR RUIDO DE FONDO ===")
    print("Este programa te permitirá grabar audio y aplicar filtros para mejorar la calidad de voz")
    
    while True:
        print("\nOpciones:")
        print("1. Grabar audio")
        print("2. Cargar audio desde archivo")
        print("3. Reproducir audio original")
        print("4. Visualizar señal en tiempo")
        print("5. Visualizar espectro de frecuencia")
        print("6. Crear espectrograma")
        print("7. Aplicar filtro pasa-banda para voz")
        print("8. Aplicar reducción de ruido adaptativa")
        print("9. Reproducir audio filtrado")
        print("10. Comparar original vs filtrado")
        print("0. Salir")
        
        choice = input("\nSelecciona una opción: ")
        
        if choice == '1':
            processor.record_audio()
        elif choice == '2':
            path = input("Ingresa la ruta del archivo de audio (por defecto: audio_original.wav): ") or "audio_original.wav"
            processor.load_audio(path)
        elif choice == '3':
            processor.play_audio()
        elif choice == '4':
            processor.visualize_time_domain()
        elif choice == '5':
            processor.visualize_frequency_domain()
        elif choice == '6':
            processor.create_spectrogram()
        elif choice == '7':
            # Se puede ajustar el rango de frecuencias según la necesidad
            lowcut = float(input("Ingresa la frecuencia de corte inferior (Hz) [recomendado 300]: ") or 300)
            highcut = float(input("Ingresa la frecuencia de corte superior (Hz) [recomendado 3400]: ") or 3400)
            filter_coeffs = processor.design_bandpass_filter(lowcut, highcut)
            processor.apply_filter(filter_coeffs)
            
            # Visualizamos el resultado
            processor.visualize_time_domain(processor.filtered_audio, "Señal Filtrada en Dominio del Tiempo")
            processor.visualize_frequency_domain(processor.filtered_audio, "Espectro de Frecuencia (Señal Filtrada)")
            processor.create_spectrogram(processor.filtered_audio, "Espectrograma (Señal Filtrada)")
        elif choice == '8':
            noise_level = float(input("Ingresa el nivel de reducción de ruido (0.1-1.0) [recomendado 0.5]: ") or 0.5)
            processor.apply_noise_reduction(noise_level)
            
            # Visualizamos el resultado
            processor.visualize_time_domain(processor.filtered_audio, "Señal con Reducción de Ruido")
            processor.visualize_frequency_domain(processor.filtered_audio, "Espectro de Frecuencia (Reducción de Ruido)")
            processor.create_spectrogram(processor.filtered_audio, "Espectrograma (Reducción de Ruido)")
        elif choice == '9':
            if processor.filtered_audio is not None:
                processor.play_audio(processor.filtered_audio)
            else:
                print("No hay audio filtrado. Primero aplica un filtro.")
        elif choice == '10':
            if processor.filtered_audio is not None:
                print("Reproduciendo audio original...")
                processor.play_audio()
                time.sleep(1)
                print("Reproduciendo audio filtrado...")
                processor.play_audio(processor.filtered_audio)
            else:
                print("No hay audio filtrado. Primero aplica un filtro.")
        elif choice == '0':
            print("¡Hasta pronto!")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()