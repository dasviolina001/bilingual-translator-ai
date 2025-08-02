from flask import Flask, request, jsonify, render_template, send_file
from translate_assamese import translate as translate_assamese
from translate_bodo import translate as translate_bodo
from flask_cors import CORS
from google.cloud import texttospeech
import os
import io

app = Flask(__name__, template_folder='templates')
CORS(app)

# Set your Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

# Initialize Google Cloud TTS client
tts_client = texttospeech.TextToSpeechClient()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()

    if not data or 'sentence' not in data or 'language' not in data:
        return jsonify({'error': 'Missing input text or language'}), 400

    text = data['sentence']
    lang = data['language']

    if lang == 'assamese':
        translated = translate_assamese(text)
    elif lang == 'bodo':
        translated = translate_bodo(text)
    else:
        return jsonify({'error': 'Unsupported language'}), 400

    return jsonify({'translated': translated})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Voice settings (adjust if Assamese or Bodo voices become available)
    voice = texttospeech.VoiceSelectionParams(
        language_code="bn-IN",  # Bengali Indian, close to Assamese/Bodo
        name="bn-IN-Chirp3-HD-Achernar"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return send_file(
        io.BytesIO(response.audio_content),
        mimetype='audio/mpeg'
    )

if __name__ == '__main__':
    app.run(debug=True)
