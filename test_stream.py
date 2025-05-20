import sounddevice as sd

fs = 44100
buffer_size = 1024

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

print("Dispositivos disponibles:")
print(sd.query_devices())

input_device = int(input("Selecciona el ID del dispositivo de entrada: "))
output_device = int(input("Selecciona el ID del dispositivo de salida: "))

input_info = sd.query_devices(input_device)
output_info = sd.query_devices(output_device)

input_channels = input_info['max_input_channels']
output_channels = output_info['max_output_channels']

print(f"Usando {input_channels} canales de entrada y {output_channels} canales de salida")

with sd.Stream(samplerate=fs,
               blocksize=buffer_size,
               device=(input_device, output_device),
               channels=(input_channels, output_channels),
               callback=callback):
    print("Procesando audio en tiempo real. Presiona Ctrl+C para detener.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Detenido.")

