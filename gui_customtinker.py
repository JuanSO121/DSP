
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import sounddevice as sd
from audio_operations import apply_equalizer
from audio_visuals import visualize_eq_response
from audio_processor import AudioProcessor
from audio_visuals import (
    visualize_time,
    visualize_frequency,
    visualize_spectrogram,
)

# Configuración del tema y apariencia
ctk.set_appearance_mode("dark")  # "dark" o "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class ModernAudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Processor Pro - Noise Reduction & EQ")
        self.root.geometry("1200x800")
        
        # Configurar el grid principal
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.processor = AudioProcessor()
        self.eq_settings_cache = None
        
        # Crear el layout principal
        self.create_layout()
        
        # Inicializar la actualización de configuraciones EQ
        self.root.after(200, self.update_eq_settings)

    def create_layout(self):
        # Frame principal con scroll
        self.main_frame = ctk.CTkScrollableFrame(self.root)
        self.main_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)
        
        # Configurar grid del frame principal
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Crear secciones
        self.create_header()
        self.create_file_section()
        self.create_playback_section()
        self.create_processing_section()
        self.create_visualization_section()
        self.create_equalizer_section()
        self.create_monitoring_section()

    def create_header(self):
        # Título principal
        title_label = ctk.CTkLabel(
            self.main_frame, 
            text="🎵 Audio Processor Pro", 
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Subtítulo
        subtitle_label = ctk.CTkLabel(
            self.main_frame, 
            text="Professional Audio Processing & Noise Reduction",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 40))

    def create_file_section(self):
        # Frame para operaciones de archivo
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        file_frame.grid_columnconfigure(0, weight=1)
        file_frame.grid_columnconfigure(1, weight=1)
        
        # Título de sección
        ctk.CTkLabel(
            file_frame, 
            text="📁 File Operations", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(15, 10))
        
        # Botones de archivo
        self.load_button = ctk.CTkButton(
            file_frame, 
            text="🎵 Load Audio File", 
            command=self.load_audio,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.load_button.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 15), sticky="ew")

    def create_playback_section(self):
        # Frame para reproducción
        playback_frame = ctk.CTkFrame(self.main_frame)
        playback_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        playback_frame.grid_columnconfigure(0, weight=1)
        playback_frame.grid_columnconfigure(1, weight=1)
        
        # Título de sección
        ctk.CTkLabel(
            playback_frame, 
            text="🔊 Playback Controls", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(15, 10))
        
        # Botones de reproducción
        self.play_original_btn = ctk.CTkButton(
            playback_frame, 
            text="▶️ Play Original", 
            command=self.play_original,
            height=35
        )
        self.play_original_btn.grid(row=1, column=0, padx=(20, 10), pady=(0, 15), sticky="ew")
        
        self.play_filtered_btn = ctk.CTkButton(
            playback_frame, 
            text="▶️ Play Processed", 
            command=self.play_filtered,
            height=35
        )
        self.play_filtered_btn.grid(row=1, column=1, padx=(10, 20), pady=(0, 15), sticky="ew")

    def create_processing_section(self):
        # Frame para procesamiento
        processing_frame = ctk.CTkFrame(self.main_frame)
        processing_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        processing_frame.grid_columnconfigure(0, weight=1)
        
        # Título de sección
        ctk.CTkLabel(
            processing_frame, 
            text="🔧 Audio Processing", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=(15, 10))
        
        # Botón de reducción de ruido
        self.reduce_noise_btn = ctk.CTkButton(
            processing_frame, 
            text="🎛️ Reduce Noise", 
            command=self.reduce_noise,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.reduce_noise_btn.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="ew")

    def create_visualization_section(self):
        # Frame para visualización
        vis_frame = ctk.CTkFrame(self.main_frame)
        vis_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        vis_frame.grid_columnconfigure(0, weight=1)
        vis_frame.grid_columnconfigure(1, weight=1)
        vis_frame.grid_columnconfigure(2, weight=1)
        
        # Título de sección
        ctk.CTkLabel(
            vis_frame, 
            text="📊 Visualization Tools", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=3, pady=(15, 10))
        
        # Botones de visualización
        self.time_vis_btn = ctk.CTkButton(
            vis_frame, 
            text="📈 Time Domain", 
            command=self.visualize_time,
            height=35
        )
        self.time_vis_btn.grid(row=1, column=0, padx=(20, 7), pady=(0, 15), sticky="ew")
        
        self.freq_vis_btn = ctk.CTkButton(
            vis_frame, 
            text="📊 Frequency", 
            command=self.visualize_frequency,
            height=35
        )
        self.freq_vis_btn.grid(row=1, column=1, padx=7, pady=(0, 15), sticky="ew")
        
        self.spec_vis_btn = ctk.CTkButton(
            vis_frame, 
            text="🌈 Spectrogram", 
            command=self.visualize_spectrogram,
            height=35
        )
        self.spec_vis_btn.grid(row=1, column=2, padx=(7, 20), pady=(0, 15), sticky="ew")

    def create_equalizer_section(self):
        # Frame principal del ecualizador
        eq_main_frame = ctk.CTkFrame(self.main_frame)
        eq_main_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        eq_main_frame.grid_columnconfigure(0, weight=1)
        eq_main_frame.grid_columnconfigure(1, weight=1)
        
        # Título de sección
        ctk.CTkLabel(
            eq_main_frame, 
            text="🎚️ 5-Band Equalizer", 
            font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(15, 20))
        
        # Frame para filtros
        filters_frame = ctk.CTkFrame(eq_main_frame)
        filters_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
        filters_frame.grid_columnconfigure(0, weight=1)
        filters_frame.grid_columnconfigure(1, weight=1)
        
        # Filtros LPF y HPF
        ctk.CTkLabel(
            filters_frame, 
            text="Low Pass Filter", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, pady=(10, 5))
        
        self.lpf_slider = ctk.CTkSlider(
            filters_frame, 
            from_=500, 
            to=8000, 
            number_of_steps=76,
            height=20
        )
        self.lpf_slider.set(4000)
        self.lpf_slider.grid(row=1, column=0, padx=(20, 10), pady=(0, 5), sticky="ew")
        
        self.lpf_label = ctk.CTkLabel(filters_frame, text="4000 Hz")
        self.lpf_label.grid(row=2, column=0, pady=(0, 10))
        
        ctk.CTkLabel(
            filters_frame, 
            text="High Pass Filter", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=1, pady=(10, 5))
        
        self.hpf_slider = ctk.CTkSlider(
            filters_frame, 
            from_=20, 
            to=1000, 
            number_of_steps=20,
            height=20
        )
        self.hpf_slider.set(200)
        self.hpf_slider.grid(row=1, column=1, padx=(10, 20), pady=(0, 5), sticky="ew")
        
        self.hpf_label = ctk.CTkLabel(filters_frame, text="200 Hz")
        self.hpf_label.grid(row=2, column=1, pady=(0, 10))
        
        # Configurar callbacks para actualizar labels
        self.lpf_slider.configure(command=self.update_lpf_label)
        self.hpf_slider.configure(command=self.update_hpf_label)
        
        # Frame para bandas paramétricas
        bands_frame = ctk.CTkFrame(eq_main_frame)
        bands_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
        
        # Título de bandas
        ctk.CTkLabel(
            bands_frame, 
            text="Parametric Bands", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=4, pady=(10, 15))
        
        # Headers para las bandas
        ctk.CTkLabel(bands_frame, text="Frequency (Hz)", font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=0, padx=10)
        ctk.CTkLabel(bands_frame, text="Gain (dB)", font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=1, padx=10)
        ctk.CTkLabel(bands_frame, text="Q Factor", font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=2, padx=10)
        ctk.CTkLabel(bands_frame, text="Values", font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=3, padx=10)
        
        # Crear bandas paramétricas
        self.bands = []
        self.band_labels = []
        
        for i in range(3):
            row = i + 2
            
            # Frequency slider
            f0 = ctk.CTkSlider(bands_frame, from_=100, to=8000, number_of_steps=79, height=15)
            f0.set(1000)
            f0.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
            
            # Gain slider
            gain = ctk.CTkSlider(bands_frame, from_=-12, to=12, number_of_steps=24, height=15)
            gain.set(0)
            gain.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
            
            # Q slider
            q = ctk.CTkSlider(bands_frame, from_=0.1, to=5, number_of_steps=49, height=15)
            q.set(1.0)
            q.grid(row=row, column=2, padx=10, pady=5, sticky="ew")
            
            # Label para mostrar valores
            label = ctk.CTkLabel(bands_frame, text="1000Hz | 0dB | Q1.0")
            label.grid(row=row, column=3, padx=10, pady=5)
            
            self.bands.append((f0, gain, q))
            self.band_labels.append(label)
            
            # Configurar callbacks para actualizar labels
            f0.configure(command=lambda val, idx=i: self.update_band_label(idx))
            gain.configure(command=lambda val, idx=i: self.update_band_label(idx))
            q.configure(command=lambda val, idx=i: self.update_band_label(idx))
        
        # Configurar columnas del frame de bandas
        for i in range(4):
            bands_frame.grid_columnconfigure(i, weight=1)
        
        # Botones del ecualizador
        eq_buttons_frame = ctk.CTkFrame(eq_main_frame)
        eq_buttons_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
        eq_buttons_frame.grid_columnconfigure(0, weight=1)
        eq_buttons_frame.grid_columnconfigure(1, weight=1)
        eq_buttons_frame.grid_columnconfigure(2, weight=1)
        
        self.eq_btn = ctk.CTkButton(
            eq_buttons_frame, 
            text="🎛️ Apply EQ", 
            command=self.apply_eq,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.eq_btn.grid(row=0, column=0, padx=(20, 7), pady=10, sticky="ew")
        
        self.show_eq_btn = ctk.CTkButton(
            eq_buttons_frame, 
            text="📈 Show EQ Curve", 
            command=self.show_eq_curve,
            height=35
        )
        self.show_eq_btn.grid(row=0, column=1, padx=7, pady=10, sticky="ew")
        
        self.play_eq_btn = ctk.CTkButton(
            eq_buttons_frame, 
            text="▶️ Play EQ Audio", 
            command=self.play_filtered,
            height=35
        )
        self.play_eq_btn.grid(row=0, column=2, padx=(7, 20), pady=10, sticky="ew")

    def create_monitoring_section(self):
        # Frame para monitoreo
        monitor_frame = ctk.CTkFrame(self.main_frame)
        monitor_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        monitor_frame.grid_columnconfigure(0, weight=1)
        monitor_frame.grid_columnconfigure(1, weight=1)
        
        # Título de sección
        ctk.CTkLabel(
            monitor_frame, 
            text="🎤 Real-time Monitoring", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(15, 10))
        
        # Botones de monitoreo
        self.monitor_btn = ctk.CTkButton(
            monitor_frame, 
            text="🎤 Start Monitoring", 
            command=self.monitor_mic,
            height=35,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.monitor_btn.grid(row=1, column=0, padx=(20, 10), pady=(0, 15), sticky="ew")
        
        self.stop_monitor_btn = ctk.CTkButton(
            monitor_frame, 
            text="⏹️ Stop Monitoring", 
            command=self.stop_monitoring,
            height=35,
            fg_color="red",
            hover_color="darkred"
        )
        self.stop_monitor_btn.grid(row=1, column=1, padx=(10, 20), pady=(0, 15), sticky="ew")

    # Funciones para actualizar labels
    def update_lpf_label(self, value):
        self.lpf_label.configure(text=f"{int(value)} Hz")
    
    def update_hpf_label(self, value):
        self.hpf_label.configure(text=f"{int(value)} Hz")
    
    def update_band_label(self, idx):
        f0, gain, q = self.bands[idx]
        freq_val = int(f0.get())
        gain_val = int(gain.get())
        q_val = round(q.get(), 1)
        self.band_labels[idx].configure(text=f"{freq_val}Hz | {gain_val:+d}dB | Q{q_val}")

    # ========== FUNCIONES PRINCIPALES (mantenidas del código original) ==========
    def load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        if not path:
            return
        
        success = self.processor.load_audio(path)
        if success is not None:
            messagebox.showinfo("Load Complete", f"Audio loaded:\n{os.path.basename(path)}")
            self.root.title(f"Audio Processor Pro - {os.path.basename(path)}")
        else:
            messagebox.showerror("Error", "Could not load audio file.")

    def play_original(self):
        if self.processor.audio_data is not None:
            sd.play(self.processor.audio_data, self.processor.fs)
            sd.wait()
        else:
            messagebox.showwarning("Warning", "Please load an audio file first.")

    def play_filtered(self):
        if self.processor.filtered_audio is not None:
            sd.play(self.processor.filtered_audio, self.processor.fs)
            sd.wait()
        else:
            messagebox.showwarning("Warning", "Please apply processing first.")

    def reduce_noise(self):
        result = self.processor.reduce_noise(level=0.5)
        if result is not None:
            messagebox.showinfo("Success", "Noise reduction applied.")
        else:
            messagebox.showwarning("Warning", "Could not apply noise reduction.")

    def visualize_time(self):
        if self.processor.audio_data is not None:
            visualize_time(self.processor.audio_data, self.processor.fs)
        else:
            messagebox.showwarning("Warning", "Please load an audio file first.")

    def visualize_frequency(self):
        if self.processor.audio_data is not None:
            visualize_frequency(self.processor.audio_data, self.processor.fs)
        else:
            messagebox.showwarning("Warning", "Please load an audio file first.")

    def visualize_spectrogram(self):
        if self.processor.audio_data is not None:
            visualize_spectrogram(self.processor.audio_data, self.processor.fs)
        else:
            messagebox.showwarning("Warning", "Please load an audio file first.")

    def apply_eq(self):
        if self.processor.audio_data is None:
            messagebox.showwarning("Warning", "Please load an audio file first.")
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
        messagebox.showinfo("Success", "Equalizer applied successfully.")

    def show_eq_curve(self):
        if self.processor.audio_data is None:
            messagebox.showwarning("Warning", "Please load an audio file first.")
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
        messagebox.showinfo("Monitoring", "Real-time microphone monitoring started.")

    def stop_monitoring(self):
        self.processor.stop_monitoring()
        messagebox.showinfo("Monitoring Stopped", "Monitoring stopped and audio saved.")

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
        except:
            self.eq_settings_cache = None
        self.root.after(200, self.update_eq_settings)

# ========== INICIO ==========
if __name__ == "__main__":
    root = ctk.CTk()
    app = ModernAudioApp(root)
    root.mainloop()