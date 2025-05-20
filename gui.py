import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sounddevice as sd
from audio_operations import apply_equalizer
#from audio_operations import apply_compressor
from audio_visuals import visualize_eq_response



from audio_processor import AudioProcessor
from audio_visuals import (
    visualize_time,
    visualize_frequency,
    visualize_spectrogram,
)

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Audio - Reducción de Ruido")
        self.processor = AudioProcessor()
        self.eq_settings_cache = None
        self.root.after(200, self.update_eq_settings)

                # ========== INTERFAZ DE BOTONES ==========

        self.load_button = tk.Button(root, text="Cargar Audio", width=20, command=self.load_audio)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.play_original_btn = tk.Button(root, text="Reproducir Original", width=20, command=self.play_original)
        self.play_original_btn.grid(row=0, column=1, padx=10, pady=10)

        self.reduce_noise_btn = tk.Button(root, text="Reducir Ruido", width=20, command=self.reduce_noise)
        self.reduce_noise_btn.grid(row=1, column=0, padx=10, pady=10)

        self.play_filtered_btn = tk.Button(root, text="Reproducir Filtrado", width=20, command=self.play_filtered)
        self.play_filtered_btn.grid(row=1, column=1, padx=10, pady=10)

        self.time_vis_btn = tk.Button(root, text="Visualizar Tiempo", width=20, command=self.visualize_time)
        self.time_vis_btn.grid(row=2, column=0, padx=10, pady=10)

        self.freq_vis_btn = tk.Button(root, text="Visualizar Frecuencia", width=20, command=self.visualize_frequency)
        self.freq_vis_btn.grid(row=2, column=1, padx=10, pady=10)

        self.spec_vis_btn = tk.Button(root, text="Espectrograma", width=20, command=self.visualize_spectrogram)
        self.spec_vis_btn.grid(row=3, column=0, padx=10, pady=10)

        self.show_eq_btn = tk.Button(root, text="Ver curva del Ecualizador", width=25, command=self.show_eq_curve)
        self.show_eq_btn.grid(row=11, column=0, columnspan=2, padx=10, pady=10)

        # ========== TÍTULO ECUALIZADOR ==========
        tk.Label(root, text="ECUALIZADOR DE 5 BANDAS", font=("Helvetica", 10, "bold")).grid(row=4, column=0, columnspan=2, pady=5)

        # LPF - Filtro FIR
        tk.Label(root, text="LPF cutoff (Hz)").grid(row=5, column=0)
        self.lpf_slider = tk.Scale(root, from_=500, to=8000, resolution=100, orient=tk.HORIZONTAL)
        self.lpf_slider.set(4000)
        self.lpf_slider.grid(row=5, column=1)

        # HPF - Filtro FIR
        tk.Label(root, text="HPF cutoff (Hz)").grid(row=6, column=0)
        self.hpf_slider = tk.Scale(root, from_=20, to=1000, resolution=50, orient=tk.HORIZONTAL)
        self.hpf_slider.set(200)
        self.hpf_slider.grid(row=6, column=1)

        # Bandas paramétricas (IIR)
        self.bands = []
        for i in range(3):
            tk.Label(root, text=f"Banda {i+3} - f0 / gain / Q").grid(row=7+i, column=0)
            f0 = tk.Scale(root, from_=100, to=8000, resolution=100, orient=tk.HORIZONTAL)
            gain = tk.Scale(root, from_=-12, to=12, resolution=1, orient=tk.HORIZONTAL)
            q = tk.Scale(root, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL)
            f0.set(1000)
            gain.set(0)
            q.set(1.0)
            f0.grid(row=7+i, column=1)
            gain.grid(row=7+i, column=2)
            q.grid(row=7+i, column=3)
            self.bands.append((f0, gain, q))

        # Botón aplicar EQ
        self.eq_btn = tk.Button(root, text="Aplicar Ecualizador", width=20, command=self.apply_eq)
        self.eq_btn.grid(row=10, column=0, padx=10, pady=10)

        # Reproducir ecualizado
        self.play_eq_btn = tk.Button(root, text="Reproducir Ecualizado", width=20, command=self.play_filtered)
        self.play_eq_btn.grid(row=10, column=1, padx=10, pady=10)

        self.monitor_btn = tk.Button(root, text="Monitorear Micrófono", width=25, command=self.monitor_mic)
        self.monitor_btn.grid(row=12, column=0, columnspan=2, pady=10)

        self.stop_monitor_btn = tk.Button(root, text="Detener Monitoreo", width=25, command=self.stop_monitoring)
        self.stop_monitor_btn.grid(row=13, column=0, columnspan=2, pady=10)

    # ========== FUNCIONES PRINCIPALES ==========

    def load_audio(self):
        path = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if not path:
            return  # Cancelado

        success = self.processor.load_audio(path)
        if success is not None:
            messagebox.showinfo("Carga completada", f"Audio cargado:\n{os.path.basename(path)}")
            self.root.title(f"Procesador de Audio - {os.path.basename(path)}")
        else:
            messagebox.showerror("Error", "No se pudo cargar el archivo de audio.")

    def play_original(self):
        if self.processor.audio_data is not None:
            sd.play(self.processor.audio_data, self.processor.fs)
            sd.wait()
        else:
            messagebox.showwarning("Aviso", "Primero debes cargar un audio.")

    def play_filtered(self):
        if self.processor.filtered_audio is not None:
            sd.play(self.processor.filtered_audio, self.processor.fs)
            sd.wait()
        else:
            messagebox.showwarning("Aviso", "Primero aplica reducción de ruido.")

    def reduce_noise(self):
        result = self.processor.reduce_noise(level=0.5)
        if result is not None:
            messagebox.showinfo("Éxito", "Reducción de ruido aplicada.")
        else:
            messagebox.showwarning("Aviso", "No se pudo aplicar reducción de ruido.")

    def visualize_time(self):
        if self.processor.audio_data is not None:
            visualize_time(self.processor.audio_data, self.processor.fs)
        else:
            messagebox.showwarning("Aviso", "Primero debes cargar un audio.")

    def visualize_frequency(self):
        if self.processor.audio_data is not None:
            visualize_frequency(self.processor.audio_data, self.processor.fs)
        else:
            messagebox.showwarning("Aviso", "Primero debes cargar un audio.")

    def visualize_spectrogram(self):
        if self.processor.audio_data is not None:
            visualize_spectrogram(self.processor.audio_data, self.processor.fs)
        else:
            messagebox.showwarning("Aviso", "Primero debes cargar un audio.")
  
    def apply_eq(self):
        if self.processor.audio_data is None:
            messagebox.showwarning("Aviso", "Primero debes cargar un audio.")
            return

        eq_settings = {
            "lpf_cutoff": self.lpf_slider.get(),
            "hpf_cutoff": self.hpf_slider.get(),
            "bands": []
        }

        for f0, gain, q in self.bands:
            eq_settings["bands"].append({
                "f0": f0.get(),
                "gain": gain.get(),
                "Q": q.get()
            })

        self.processor.filtered_audio = apply_equalizer(self.processor.audio_data, self.processor.fs, eq_settings)
        messagebox.showinfo("Éxito", "Ecualizador aplicado correctamente.")

    def show_eq_curve(self):
        if self.processor.audio_data is None:
            messagebox.showwarning("Aviso", "Primero debes cargar un audio.")
            return

        eq_settings = {
            "lpf_cutoff": self.lpf_slider.get(),
            "hpf_cutoff": self.hpf_slider.get(),
            "bands": []
        }

        for f0, gain, q in self.bands:
            eq_settings["bands"].append({
                "f0": f0.get(),
                "gain": gain.get(),
                "Q": q.get()
            })

        visualize_eq_response(self.processor.fs, eq_settings)

    def monitor_mic(self):
        def get_eq_settings():
            return self.eq_settings_cache
        self.processor.monitor_audio(get_eq_settings)
        messagebox.showinfo("Monitoreo", "Escuchando el micrófono en tiempo real.")


    def stop_monitoring(self):
        self.processor.stop_monitoring()
        messagebox.showinfo("Monitoreo detenido", "Se ha detenido el monitoreo y el audio fue guardado.")


    def update_eq_settings(self):
        try:
            self.eq_settings_cache = {
                "lpf_cutoff": self.lpf_slider.get(),
                "hpf_cutoff": self.hpf_slider.get(),
                "bands": [
                    {
                        "f0": f0.get(),
                        "gain": gain.get(),
                        "Q": q.get()
                    } for f0, gain, q in self.bands
                ]
            }
        except tk.TclError:
            self.eq_settings_cache = None
        self.root.after(200, self.update_eq_settings)





# ========== INICIO ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
