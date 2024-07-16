from flask import Flask, request, render_template, jsonify
import os
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

@app.route('/hey_akbank')
def hey_akbank():
    return render_template('hey_akbank.html')

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

@app.route('/process-speech', methods=['POST'])
def process_speech():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    audio_file = request.files['file']
    audio_path = "temp_speech.wav"
    audio_file.save(audio_path)

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

@app.route('/process-target-speech', methods=['POST'])
def process_target_speech():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    audio_file = request.files['file']
    audio_path = f"model_training/voice_samples/target_speaker/{uuid.uuid4()}.wav"
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="tr-TR")
            return jsonify({'text': text, 'success': True})
        except sr.UnknownValueError:
            return jsonify({'error': 'Speech was unintelligible', 'success': False})
        except sr.RequestError:
            return jsonify({'error': 'Could not request results from Google Speech Recognition service', 'success': False})

@app.route('/process-wakeword-speech', methods=['POST'])
def process_wakeword_speech():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    audio_file = request.files['file']
    audio_path = "temp_wakeword_speech.wav"
    audio_file.save(audio_path)

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

if __name__ == '__main__':
    app.run(debug=True)
