import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import queue
import threading
from scipy import signal

class AudioProcessor:
    def __init__(self):
        self.audio_data = None
        self.filtered_audio = None
        self.fs = None
        self.is_monitoring = False
        self.monitor_thread = None
        self.audio_buffer = None
        self.buffer_size = 1024  # Tamaño del bloque para streaming
        self.overlap = 256       # Solapamiento entre bloques
        self.eq_processor = None
        self.recording_buffer = []  # Para guardar el audio procesado
        
    def load_audio(self, filepath):
        """Carga un archivo de audio"""
        try:
            self.audio_data, self.fs = sf.read(filepath)
            
            # Convertir a mono si es estéreo
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
                
            # Normalizar el audio
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
            
            # Reset del audio filtrado
            self.filtered_audio = None
            
            return self.audio_data
        except Exception as e:
            print(f"Error al cargar el audio: {e}")
            return None
    
    def reduce_noise(self, level=0.5):
        """Aplica reducción de ruido simple"""
        from audio_operations import apply_noise_reduction
        try:
            if self.audio_data is not None:
                self.filtered_audio = apply_noise_reduction(self.audio_data, self.fs, level)
                return self.filtered_audio
            return None
        except Exception as e:
            print(f"Error en reducción de ruido: {e}")
            return None
            
    def monitor_audio(self, get_eq_settings_callback):
        """Monitorea el audio del micrófono y aplica procesamiento en tiempo real"""
        if self.is_monitoring:
            print("Ya está monitoreando")
            return
            
        self.is_monitoring = True
        self.recording_buffer = []  # Reiniciar buffer de grabación
        
        # Crear objetos para procesamiento
        from audio_operations import EQProcessor
        
        # Configuración de audio
        fs = 44100  # Frecuencia de muestreo
        self.fs = fs
        channels = 1  # Mono
        
        # Inicializar procesador EQ
        self.eq_processor = EQProcessor(fs, self.buffer_size)
        
        # Cola para comunicación entre callbacks
        q = queue.Queue(maxsize=10)
        
        # Configuración de streaming
        stream_buffer = np.zeros(self.buffer_size + self.overlap)
        output_buffer = np.zeros(self.buffer_size)
        
        def audio_callback(indata, outdata, frames, time, status):
            """Callback para el streaming de audio"""
            if status:
                print(f"Error en stream: {status}")
                
            # Obtener configuración actual del EQ
            current_eq_settings = get_eq_settings_callback()
            if current_eq_settings:
                self.eq_processor.configure(current_eq_settings)
            
            # Procesar audio de entrada (mono)
            input_mono = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
            
            # Desplazar buffer y añadir nuevos datos
            stream_buffer[:-frames] = stream_buffer[frames:]
            stream_buffer[-frames:] = input_mono
            
            # Procesar bloque completo con EQ
            processed = self.eq_processor.process_block(stream_buffer[-self.buffer_size:])
            
            # Añadir al buffer de grabación
            self.recording_buffer.append(processed.copy())
            
            # Enviar a la salida (duplicar para estéreo)
            if outdata.shape[1] > 1:
                outdata[:] = np.column_stack([processed, processed])
            else:
                outdata[:] = processed.reshape(-1, 1)
                
            # Enviar a la cola para visualización si es necesario
            try:
                q.put_nowait(processed)
            except queue.Full:
                pass
        
        # Iniciar stream
        try:
            self.audio_stream = sd.Stream(
                samplerate=fs,
                blocksize=self.buffer_size,
                channels=2,  # Estéreo para salida
                dtype='float32',
                callback=audio_callback
            )
            
            self.audio_stream.start()
            
            # Iniciar thread para grabación
            self.monitor_thread = threading.Thread(target=self._monitor_worker, args=(q,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
        except Exception as e:
            self.is_monitoring = False
            print(f"Error al iniciar monitoreo: {e}")
    
    def _monitor_worker(self, q):
        """Worker thread para monitoreo"""
        try:
            while self.is_monitoring:
                time.sleep(0.1)  # Pequeña pausa para evitar consumo excesivo de CPU
        except Exception as e:
            print(f"Error en monitor worker: {e}")
        finally:
            # Limpiar al finalizar
            while not q.empty():
                q.get()
    
    def stop_monitoring(self):
        """Detiene el monitoreo y guarda el audio procesado"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        
        try:
            # Detener stream
            self.audio_stream.stop()
            self.audio_stream.close()
            
            # Esperar a que el thread termine
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
            
            # Guardar audio procesado
            if self.recording_buffer:
                # Convertir lista de buffers a un array continuo
                recorded_audio = np.concatenate(self.recording_buffer)
                
                # Normalizar
                max_amp = np.max(np.abs(recorded_audio))
                if max_amp > 0:
                    recorded_audio = recorded_audio / max_amp * 0.9
                
                # Guardar como WAV
                sf.write('audio_procesado.wav', recorded_audio, self.fs)
                
                # También guardarlo como audio filtrado para reproducción
                self.filtered_audio = recorded_audio
                
        except Exception as e:
            print(f"Error al detener monitoreo: {e}")