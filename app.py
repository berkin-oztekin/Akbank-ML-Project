from flask import Flask, request, render_template, jsonify
import os
import wave
import io
import uuid
import speech_recognition as sr
import joblib
from utils.feature_extraction import extract_features

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_training/speaker_recognition_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    recognizer = sr.Recognizer()
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    features = extract_features(audio_path)
    speaker_prob = model.predict_proba([features])[0][1]

    if speaker_prob < 0.7:
        return jsonify({'error': 'Speaker not recognized'})

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="tr-TR")
            return jsonify({'text': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Speech was unintelligible'})
        except sr.RequestError:
            return jsonify({'error': 'Could not request results from Google Speech Recognition service'})


@app.route('/record-speech-to-text', methods=['POST'])
def record_speech_to_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Read the audio file from the request
    audio_data = audio_file.read()
    audio_stream = io.BytesIO(audio_data)

    # Ensure the file is in WAV format and use it directly
    try:
        with wave.open(audio_stream, 'rb') as wf:
            print(f"Channels: {wf.getnchannels()}, Sample Width: {wf.getsampwidth()}, Frame Rate: {wf.getframerate()}")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [16000, 44100, 48000]:
                return jsonify({'error': 'Unsupported audio format'})

            audio_path = "temp_audio.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
    except wave.Error as e:
        print(f"Wave Error: {e}")
        return jsonify({'error': 'File is not in WAV format'})

    recognizer = sr.Recognizer()

    features = extract_features(audio_path)
    speaker_prob = model.predict_proba([features])[0][1]

    if speaker_prob < 0.7:
        return jsonify({'error': 'Speaker not recognized'})

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="tr-TR")
            return jsonify({'text': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Speech was unintelligible'})
        except sr.RequestError:
            return jsonify({'error': 'Could not request results from Google Speech Recognition service'})


@app.route('/record-target-speaker', methods=['POST'])
def record_target_speaker():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    target_dir = 'model_training/voice_samples/target_speaker/'
    os.makedirs(target_dir, exist_ok=True)

    # Generate a random filename
    random_filename = f"{uuid.uuid4()}.wav"
    target_path = os.path.join(target_dir, random_filename)

    try:
        with open(target_path, 'wb') as f:
            f.write(audio_file.read())
        return jsonify({'success': True, 'filename': random_filename})
    except Exception as e:
        print(f"Error saving target speaker audio: {e}")
        return jsonify({'error': 'Error saving target speaker audio'})


if __name__ == '__main__':
    app.run(debug=True)
