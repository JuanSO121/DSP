from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import io
import base64
import soundfile as sf
import numpy as np
from werkzeug.utils import secure_filename

# Importa tus módulos existentes
from audio_processor import AudioProcessor
from audio_operations import apply_equalizer
from audio_visuals import visualize_time, visualize_frequency, visualize_spectrogram, visualize_eq_response

app = Flask(__name__)
CORS(app)  # Permite requests desde el frontend

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Instancia global del procesador
processor = AudioProcessor()

# HTML template embebido
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Procesador de Audio - Reducción de Ruido</title>

</head>
<body>
    <div id="app">
        <!-- Header -->
        <header class="header">
            <h1 class="title">Procesador de Audio</h1>
            <p class="subtitle">Reducción de Ruido y Ecualización</p>
        </header>

        <!-- Main Content -->
        <main class="main-content">
            <!-- File Status -->
            <section class="file-status">
                <div class="status-indicator" id="fileStatus">
                    <span class="status-icon">📁</span>
                    <span class="status-text">No hay archivo cargado</span>
                </div>
            </section>

            <!-- Audio Controls Section -->
            <section class="audio-controls">
                <h2 class="section-title">Controles de Audio</h2>
                <div class="controls-grid">
                    <button class="btn btn-primary" id="loadAudio">
                        <span class="btn-icon">📂</span>
                        Cargar Audio
                    </button>
                    <button class="btn btn-secondary" id="playOriginal" disabled>
                        <span class="btn-icon">▶️</span>
                        Reproducir Original
                    </button>
                    <button class="btn btn-accent" id="reduceNoise" disabled>
                        <span class="btn-icon">🔧</span>
                        Reducir Ruido
                    </button>
                    <button class="btn btn-secondary" id="playFiltered" disabled>
                        <span class="btn-icon">🔊</span>
                        Reproducir Filtrado
                    </button>
                </div>
            </section>

            <!-- Visualization Section -->
            <section class="visualization">
                <h2 class="section-title">Visualización</h2>
                <div class="vis-controls controls-grid">
                    <button class="btn btn-outline" id="visualizeTime" disabled>
                        <span class="btn-icon">📈</span>
                        Visualizar Tiempo
                    </button>
                    <button class="btn btn-outline" id="visualizeFreq" disabled>
                        <span class="btn-icon">📊</span>
                        Visualizar Frecuencia
                    </button>
                    <button class="btn btn-outline" id="visualizeSpec" disabled>
                        <span class="btn-icon">🌈</span>
                        Espectrograma
                    </button>
                </div>
            </section>

            <!-- Equalizer Section -->
            <section class="equalizer">
                <h2 class="section-title">Ecualizador de 5 Bandas</h2>
                
                <!-- Filter Controls -->
                <div class="filter-section">
                    <h3 class="section-title">Filtros</h3>
                    <div class="controls-grid">
                        <div class="filter-control">
                            <label class="control-label" for="lpfSlider">
                                Filtro Pasa Bajos (Hz)
                                <span class="value-display" id="lpfValue">4000</span>
                            </label>
                            <input type="range" id="lpfSlider" class="slider" 
                                   min="500" max="8000" step="100" value="4000">
                        </div>
                        
                        <div class="filter-control">
                            <label class="control-label" for="hpfSlider">
                                Filtro Pasa Altos (Hz)
                                <span class="value-display" id="hpfValue">200</span>
                            </label>
                            <input type="range" id="hpfSlider" class="slider" 
                                   min="20" max="1000" step="50" value="200">
                        </div>
                    </div>
                </div>

                <!-- Parametric Bands -->
                <div class="bands-section">
                    <h3 class="section-title">Bandas Paramétricas</h3>
                    <div class="bands-container" id="bandsContainer">
                        <!-- Band 3 -->
                        <div class="band" data-band="3">
                            <h4>Banda 3</h4>
                            <div class="band-controls">
                                <div class="band-control">
                                    <label class="control-label">
                                        Frecuencia (Hz)
                                        <span class="value-display" id="band3FreqValue">1000</span>
                                    </label>
                                    <input type="range" id="band3Freq" class="slider" 
                                           min="100" max="8000" step="100" value="1000">
                                </div>
                                <div class="band-control">
                                    <label class="control-label">
                                        Ganancia (dB)
                                        <span class="value-display" id="band3GainValue">0</span>
                                    </label>
                                    <input type="range" id="band3Gain" class="slider" 
                                           min="-12" max="12" step="1" value="0">
                                </div>
                                <div class="band-control">
                                    <label class="control-label">
                                        Factor Q
                                        <span class="value-display" id="band3QValue">1.0</span>
                                    </label>
                                    <input type="range" id="band3Q" class="slider" 
                                           min="0.1" max="5" step="0.1" value="1.0">
                                </div>
                            </div>
                        </div>

                        <!-- Band 4 -->
                        <div class="band" data-band="4">
                            <h4>Banda 4</h4>
                            <div class="band-controls">
                                <div class="band-control">
                                    <label class="control-label">
                                        Frecuencia (Hz)
                                        <span class="value-display" id="band4FreqValue">1000</span>
                                    </label>
                                    <input type="range" id="band4Freq" class="slider" 
                                           min="100" max="8000" step="100" value="1000">
                                </div>
                                <div class="band-control">
                                    <label class="control-label">
                                        Ganancia (dB)
                                        <span class="value-display" id="band4GainValue">0</span>
                                    </label>
                                    <input type="range" id="band4Gain" class="slider" 
                                           min="-12" max="12" step="1" value="0">
                                </div>
                                <div class="band-control">
                                    <label class="control-label">
                                        Factor Q
                                        <span class="value-display" id="band4QValue">1.0</span>
                                    </label>
                                    <input type="range" id="band4Q" class="slider" 
                                           min="0.1" max="5" step="0.1" value="1.0">
                                </div>
                            </div>
                        </div>

                        <!-- Band 5 -->
                        <div class="band" data-band="5">
                            <h4>Banda 5</h4>
                            <div class="band-controls">
                                <div class="band-control">
                                    <label class="control-label">
                                        Frecuencia (Hz)
                                        <span class="value-display" id="band5FreqValue">1000</span>
                                    </label>
                                    <input type="range" id="band5Freq" class="slider" 
                                           min="100" max="8000" step="100" value="1000">
                                </div>
                                <div class="band-control">
                                    <label class="control-label">
                                        Ganancia (dB)
                                        <span class="value-display" id="band5GainValue">0</span>
                                    </label>
                                    <input type="range" id="band5Gain" class="slider" 
                                           min="-12" max="12" step="1" value="0">
                                </div>
                                <div class="band-control">
                                    <label class="control-label">
                                        Factor Q
                                        <span class="value-display" id="band5QValue">1.0</span>
                                    </label>
                                    <input type="range" id="band5Q" class="slider" 
                                           min="0.1" max="5" step="0.1" value="1.0">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- EQ Actions -->
                <div class="eq-actions controls-grid">
                    <button class="btn btn-primary" id="applyEQ" disabled>
                        <span class="btn-icon">🎛️</span>
                        Aplicar Ecualizador
                    </button>
                    <button class="btn btn-secondary" id="playEQ" disabled>
                        <span class="btn-icon">🔊</span>
                        Reproducir Ecualizado
                    </button>
                    <button class="btn btn-outline" id="showEQCurve" disabled>
                        <span class="btn-icon">📈</span>
                        Ver Curva del Ecualizador
                    </button>
                </div>
            </section>

            <!-- Real-time Monitoring Section -->
            <section class="monitoring">
                <h2 class="section-title">Monitoreo en Tiempo Real</h2>
                <div class="monitoring-controls controls-grid">
                    <button class="btn btn-accent" id="monitorMic">
                        <span class="btn-icon">🎤</span>
                        Monitorear Micrófono
                    </button>
                    <button class="btn btn-danger" id="stopMonitor" disabled>
                        <span class="btn-icon">⏹️</span>
                        Detener Monitoreo
                    </button>
                </div>
                <div class="monitoring-status" id="monitoringStatus">
                    <span class="status-icon">⚪</span>
                    <span class="status-text">Monitoreo inactivo</span>
                </div>
            </section>
        </main>

        <!-- Notifications -->
        <div class="notifications" id="notifications"></div>
    </div>

    <script>
        // API Configuration
        const API_BASE = 'http://127.0.0.1:5000/api';

        // State management
        let audioLoaded = false;
        let isMonitoring = false;
        let currentFileName = '';

        // DOM elements
        const elements = {
            fileStatus: document.getElementById('fileStatus'),
            loadAudio: document.getElementById('loadAudio'),
            playOriginal: document.getElementById('playOriginal'),
            reduceNoise: document.getElementById('reduceNoise'),
            playFiltered: document.getElementById('playFiltered'),
            visualizeTime: document.getElementById('visualizeTime'),
            visualizeFreq: document.getElementById('visualizeFreq'),
            visualizeSpec: document.getElementById('visualizeSpec'),
            applyEQ: document.getElementById('applyEQ'),
            playEQ: document.getElementById('playEQ'),
            showEQCurve: document.getElementById('showEQCurve'),
            monitorMic: document.getElementById('monitorMic'),
            stopMonitor: document.getElementById('stopMonitor'),
            monitoringStatus: document.getElementById('monitoringStatus'),
            notifications: document.getElementById('notifications')
        };

        // API Helper functions
        async function apiCall(endpoint, options = {}) {
            try {
                const response = await fetch(`${API_BASE}${endpoint}`, {
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    },
                    ...options
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Error en la API');
                }
                
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                showNotification(error.message, 'error');
                throw error;
            }
        }

        function playAudioFromBase64(base64Data) {
            const audio = new Audio(`data:audio/wav;base64,${base64Data}`);
            audio.play().catch(err => {
                showNotification('Error al reproducir audio', 'error');
                console.error('Audio playback error:', err);
            });
        }

        function showImageFromBase64(base64Data, title = 'Visualización') {
            const newWindow = window.open('', '_blank');
            newWindow.document.write(`
                <html>
                    <head><title>${title}</title></head>
                    <body style="margin:0; text-align:center; background:#f0f0f0;">
                        <img src="data:image/png;base64,${base64Data}" style="max-width:100%; height:auto;">
                    </body>
                </html>
            `);
        }

        // Notification system
        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.innerHTML = `
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.remove()">×</button>
            `;
            elements.notifications.appendChild(notification);
            
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 5000);
        }

        // File status update
        function updateFileStatus(fileName = null) {
            const statusIcon = elements.fileStatus.querySelector('.status-icon');
            const statusText = elements.fileStatus.querySelector('.status-text');
            
            if (fileName) {
                statusIcon.textContent = '✅';
                statusText.textContent = `Archivo cargado: ${fileName}`;
                elements.fileStatus.classList.add('loaded');
            } else {
                statusIcon.textContent = '📁';
                statusText.textContent = 'No hay archivo cargado';
                elements.fileStatus.classList.remove('loaded');
            }
        }

        // Enable/disable buttons based on state
        function updateButtonStates() {
            const audioButtons = [
                elements.playOriginal, elements.reduceNoise, elements.visualizeTime,
                elements.visualizeFreq, elements.visualizeSpec, elements.applyEQ,
                elements.showEQCurve
            ];
            
            audioButtons.forEach(btn => {
                btn.disabled = !audioLoaded;
            });
        }

        // Get EQ settings
        function getEQSettings() {
            const settings = {
                lpf_cutoff: parseInt(document.getElementById('lpfSlider').value),
                hpf_cutoff: parseInt(document.getElementById('hpfSlider').value),
                bands: []
            };

            for (let i = 3; i <= 5; i++) {
                settings.bands.push({
                    f0: parseInt(document.getElementById(`band${i}Freq`).value),
                    gain: parseInt(document.getElementById(`band${i}Gain`).value),
                    Q: parseFloat(document.getElementById(`band${i}Q`).value)
                });
            }

            return settings;
        }

        // Slider value updates
        function updateSliderValues() {
            // LPF and HPF
            const lpfSlider = document.getElementById('lpfSlider');
            const hpfSlider = document.getElementById('hpfSlider');
            
            lpfSlider.addEventListener('input', () => {
                document.getElementById('lpfValue').textContent = lpfSlider.value;
            });
            
            hpfSlider.addEventListener('input', () => {
                document.getElementById('hpfValue').textContent = hpfSlider.value;
            });

            // Parametric bands
            for (let i = 3; i <= 5; i++) {
                const freqSlider = document.getElementById(`band${i}Freq`);
                const gainSlider = document.getElementById(`band${i}Gain`);
                const qSlider = document.getElementById(`band${i}Q`);

                freqSlider.addEventListener('input', () => {
                    document.getElementById(`band${i}FreqValue`).textContent = freqSlider.value;
                });

                gainSlider.addEventListener('input', () => {
                    document.getElementById(`band${i}GainValue`).textContent = gainSlider.value;
                });

                qSlider.addEventListener('input', () => {
                    document.getElementById(`band${i}QValue`).textContent = qSlider.value;
                });
            }
        }

        // Event listeners
        elements.loadAudio.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.wav,.mp3,.flac';
            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (file) {
                    showNotification('Cargando archivo...', 'info');
                    
                    const formData = new FormData();
                    formData.append('audio', file);
                    
                    try {
                        const response = await fetch(`${API_BASE}/upload`, {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            currentFileName = result.filename;
                            audioLoaded = true;
                            updateFileStatus(currentFileName);
                            updateButtonStates();
                            showNotification(result.message, 'success');
                            document.title = `Procesador de Audio - ${currentFileName}`;
                        } else {
                            throw new Error(result.error);
                        }
                    } catch (error) {
                        showNotification(`Error al cargar archivo: ${error.message}`, 'error');
                    }
                }
            };
            input.click();
        });

        elements.playOriginal.addEventListener('click', async () => {
            showNotification('Reproduciendo audio original...', 'info');
            try {
                const result = await apiCall('/play/original');
                playAudioFromBase64(result.audio_data);
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.reduceNoise.addEventListener('click', async () => {
            showNotification('Aplicando reducción de ruido...', 'info');
            try {
                const result = await apiCall('/reduce-noise', {
                    method: 'POST',
                    body: JSON.stringify({ level: 0.5 })
                });
                
                if (result.success) {
                    elements.playFiltered.disabled = false;
                    elements.playEQ.disabled = false;
                    showNotification(result.message, 'success');
                }
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.playFiltered.addEventListener('click', async () => {
            showNotification('Reproduciendo audio filtrado...', 'info');
            try {
                const result = await apiCall('/play/filtered');
                playAudioFromBase64(result.audio_data);
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.visualizeTime.addEventListener('click', async () => {
            showNotification('Generando visualización temporal...', 'info');
            try {
                const result = await apiCall('/visualize/time');
                showImageFromBase64(result.image, 'Visualización Temporal');
                showNotification('Visualización generada', 'success');
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.visualizeFreq.addEventListener('click', async () => {
            showNotification('Generando visualización de frecuencia...', 'info');
            try {
                const result = await apiCall('/visualize/frequency');
                showImageFromBase64(result.image, 'Visualización de Frecuencia');
                showNotification('Visualización generada', 'success');
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.visualizeSpec.addEventListener('click', async () => {
            showNotification('Generando espectrograma...', 'info');
            try {
                const result = await apiCall('/visualize/spectrogram');
                showImageFromBase64(result.image, 'Espectrograma');
                showNotification('Espectrograma generado', 'success');
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.applyEQ.addEventListener('click', async () => {
            const settings = getEQSettings();
            showNotification('Aplicando ecualizador...', 'info');
            try {
                const result = await apiCall('/equalizer/apply', {
                    method: 'POST',
                    body: JSON.stringify(settings)
                });
                
                if (result.success) {
                    elements.playEQ.disabled = false;
                    showNotification(result.message, 'success');
                }
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.playEQ.addEventListener('click', async () => {
            showNotification('Reproduciendo audio ecualizado...', 'info');
            try {
                const result = await apiCall('/play/filtered');
                playAudioFromBase64(result.audio_data);
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.showEQCurve.addEventListener('click', async () => {
            const settings = getEQSettings();
            showNotification('Generando curva del ecualizador...', 'info');
            try {
                const result = await apiCall('/equalizer/curve', {
                    method: 'POST',
                    body: JSON.stringify(settings)
                });
                showImageFromBase64(result.image, 'Curva del Ecualizador');
                showNotification('Curva generada', 'success');
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.monitorMic.addEventListener('click', async () => {
            const settings = getEQSettings();
            showNotification('Iniciando monitoreo del micrófono...', 'info');
            
            try {
                const result = await apiCall('/monitor/start', {
                    method: 'POST',
                    body: JSON.stringify(settings)
                });
                
                if (result.success) {
                    isMonitoring = true;
                    elements.monitorMic.disabled = true;
                    elements.stopMonitor.disabled = false;
                    
                    const statusIcon = elements.monitoringStatus.querySelector('.status-icon');
                    const statusText = elements.monitoringStatus.querySelector('.status-text');
                    statusIcon.textContent = '🔴';
                    statusText.textContent = 'Monitoreando micrófono...';
                    elements.monitoringStatus.classList.add('monitoring');
                    
                    showNotification(result.message, 'success');
                }
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        elements.stopMonitor.addEventListener('click', async () => {
            showNotification('Deteniendo monitoreo...', 'info');
            
            try {
                const result = await apiCall('/monitor/stop', {
                    method: 'POST'
                });
                
                if (result.success) {
                    isMonitoring = false;
                    elements.monitorMic.disabled = false;
                    elements.stopMonitor.disabled = true;
                    
                    const statusIcon = elements.monitoringStatus.querySelector('.status-icon');
                    const statusText = elements.monitoringStatus.querySelector('.status-text');
                    statusIcon.textContent = '⚪';
                    statusText.textContent = 'Monitoreo inactivo';
                    elements.monitoringStatus.classList.remove('monitoring');
                    
                    showNotification(result.message, 'success');
                }
            } catch (error) {
                // Error already handled in apiCall
            }
        });

        // Initialize
        updateSliderValues();
        updateButtonStates();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Servir la página principal"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Endpoint para cargar archivos de audio"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se encontró archivo de audio'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Cargar el audio en el procesador
            success = processor.load_audio(filepath)
            
            if success is not None:
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'message': f'Audio cargado: {filename}',
                    'sample_rate': processor.fs,
                    'duration': len(processor.audio_data) / processor.fs
                })
            else:
                return jsonify({'error': 'No se pudo procesar el archivo de audio'}), 500
        
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/play/original', methods=['GET'])
def play_original():
    """Endpoint para reproducir audio original"""
    try:
        if processor.audio_data is None:
            return jsonify({'error': 'No hay audio cargado'}), 400
        
        # Convertir audio a base64 para enviarlo al frontend
        buffer = io.BytesIO()
        sf.write(buffer, processor.audio_data, processor.fs, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'audio_data': audio_base64,
            'sample_rate': processor.fs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/play/filtered', methods=['GET'])
def play_filtered():
    """Endpoint para reproducir audio filtrado"""
    try:
        if processor.filtered_audio is None:
            return jsonify({'error': 'No hay audio filtrado disponible'}), 400
        
        buffer = io.BytesIO()
        sf.write(buffer, processor.filtered_audio, processor.fs, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'audio_data': audio_base64,
            'sample_rate': processor.fs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reduce-noise', methods=['POST'])
def reduce_noise():
    """Endpoint para aplicar reducción de ruido"""
    try:
        if processor.audio_data is None:
            return jsonify({'error': 'No hay audio cargado'}), 400
        
        data = request.get_json()
        level = data.get('level', 0.5)
        
        result = processor.reduce_noise(level=level)
        
        if result is not None:
            return jsonify({
                'success': True,
                'message': 'Reducción de ruido aplicada correctamente'
            })
        else:
            return jsonify({'error': 'No se pudo aplicar la reducción de ruido'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/equalizer/apply', methods=['POST'])
def apply_eq():
    """Endpoint para aplicar ecualizador"""
    try:
        if processor.audio_data is None:
            return jsonify({'error': 'No hay audio cargado'}), 400
        
        eq_settings = request.get_json()
        
        # Aplicar ecualizador
        processor.filtered_audio = apply_equalizer(
            processor.audio_data, 
            processor.fs, 
            eq_settings
        )
        
        return jsonify({
            'success': True,
            'message': 'Ecualizador aplicado correctamente'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize/<viz_type>', methods=['GET'])
def visualize(viz_type):
    """Endpoint para generar visualizaciones"""
    try:
        if processor.audio_data is None:
            return jsonify({'error': 'No hay audio cargado'}), 400
        
        # Crear visualización según el tipo
        if viz_type == 'time':
            fig = visualize_time(processor.audio_data, processor.fs)
        elif viz_type == 'frequency':
            fig = visualize_frequency(processor.audio_data, processor.fs)
        elif viz_type == 'spectrogram':
            fig = visualize_spectrogram(processor.audio_data, processor.fs)
        else:
            return jsonify({'error': 'Tipo de visualización no válido'}), 400
        
        # Convertir figura a base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/equalizer/curve', methods=['POST'])
def show_eq_curve():
    """Endpoint para mostrar curva del ecualizador"""
    try:
        if processor.fs is None:
            return jsonify({'error': 'No hay audio cargado'}), 400
        
        eq_settings = request.get_json()
        
        # Generar curva del ecualizador
        fig = visualize_eq_response(processor.fs, eq_settings)
        
        # Convertir figura a base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/start', methods=['POST'])
def start_monitoring():
    """Endpoint para iniciar monitoreo del micrófono"""
    try:
        eq_settings = request.get_json()
        
        # Función para obtener configuración EQ
        def get_eq_settings():
            return eq_settings
        
        processor.monitor_audio(get_eq_settings)
        
        return jsonify({
            'success': True,
            'message': 'Monitoreo iniciado correctamente'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/stop', methods=['POST'])
def stop_monitoring():
    """Endpoint para detener monitoreo"""
    try:
        processor.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Monitoreo detenido y audio guardado'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint para obtener estado actual"""
    return jsonify({
        'audio_loaded': processor.audio_data is not None,
        'filtered_available': processor.filtered_audio is not None,
        'sample_rate': processor.fs if processor.fs else None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)