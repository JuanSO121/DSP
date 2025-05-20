from audio_processor import AudioProcessor
import audio_operations as ops
import audio_visuals as vis

def main():
    processor = AudioProcessor()

    while True:
        print("\nOpciones:")
        print("1. Grabar audio")
        print("2. Cargar audio")
        print("3. Reproducir audio")
        print("4. Visualizar en tiempo")
        print("5. Visualizar espectro")
        print("6. Visualizar espectrograma")
        print("7. Filtro pasa-banda")
        print("8. Reducción de ruido")
        print("0. Salir")

        choice = input("Selecciona una opción: ")

        if choice == "1":
            processor.record_audio()
        elif choice == "2":
            path = input("Ruta del archivo: ") or "audio_original.wav"
            processor.load_audio(path)
        elif choice == "3":
            processor.play_audio()
        elif choice == "4":
            vis.visualize_time(processor.audio_data, processor.fs)
        elif choice == "5":
            vis.visualize_frequency(processor.audio_data, processor.fs)
        elif choice == "6":
            vis.visualize_spectrogram(processor.audio_data, processor.fs)
        elif choice == "7":
            b = ops.design_bandpass_filter(processor.fs)
            processor.filtered_audio = ops.apply_filter(processor.audio_data, b)
        elif choice == "8":
            processor.filtered_audio = ops.apply_noise_reduction(processor.audio_data, processor.fs)
        elif choice == "0":
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()
