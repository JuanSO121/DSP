import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from audio_operations import apply_noise_reduction, apply_equalizer


class AudioProcessor:
    def __init__(self, fs=44100, duration=5):
        self.fs = fs
        self.duration = duration
        self.audio_data = None
        self.filtered_audio = None
        self._stream = None
        self._stream_thread = None
        self._recorded = []
        self._stop_monitor = threading.Event()

    def record_audio(self):
        print(f"Grabando audio por {self.duration} segundos...")
        self.audio_data = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()
        self.audio_data = self.audio_data.flatten()
        sf.write('audio_original.wav', self.audio_data, self.fs)
        print("Grabación completada y guardada como 'audio_original.wav'")
        return self.audio_data

    def load_audio(self, file_path):
        try:
            self.audio_data, new_fs = sf.read(file_path, always_2d=False)
            self.fs = new_fs
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
            print(f"Audio cargado desde {file_path}")
            return self.audio_data
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None

    def play_audio(self, audio=None):
        if audio is None:
            audio = self.audio_data
        if audio is not None:
            sd.play(audio, self.fs)
            sd.wait()
        else:
            print("No hay audio para reproducir.")

    def reduce_noise(self, level=0.5):
        if self.audio_data is not None:
            self.filtered_audio = apply_noise_reduction(self.audio_data, self.fs, level)
            return self.filtered_audio
        else:
            print("No hay audio cargado para reducir ruido.")
            return None

    def monitor_audio(self, eq_settings_getter, record=True):
        self._recorded.clear()
        self._stop_monitor.clear()

        def callback(indata, outdata, frames, time, status):
            if status:
                print("Status:", status)
            audio = indata[:, 0]

            eq_settings = None
            try:
                if eq_settings_getter:
                    eq_settings = eq_settings_getter()
            except Exception as e:
                print("Error obteniendo EQ settings:", e)

            if eq_settings:
                audio = apply_equalizer(audio, self.fs, eq_settings)

            outdata[:, 0] = audio

            if record:
                self._recorded.append(audio.copy())

            if self._stop_monitor.is_set():
                raise sd.CallbackStop()

        self._stream = sd.Stream(
            samplerate=self.fs,
            blocksize=4096,
            latency='high',
            dtype='float32',
            channels=1,
            callback=callback
        )

        def run_stream():
            with self._stream:
                sd.sleep(100000)

        self._stream_thread = threading.Thread(target=run_stream)
        self._stream_thread.daemon = True
        self._stream_thread.start()
        print("Monitoreo iniciado.")

    def stop_monitoring(self):
        self._stop_monitor.set()
        print("Monitoreo detenido.")
        if self._recorded:
            audio_full = np.concatenate(self._recorded)
            sf.write('audio_monitoreado.wav', audio_full, self.fs)
            print("Audio guardado como 'audio_monitoreado.wav'")
