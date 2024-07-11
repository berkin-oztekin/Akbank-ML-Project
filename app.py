from flask import Flask, request, render_template, jsonify
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

    # Creates an instance of the Recognizer class from the speech_recognition library
    recognizer = sr.Recognizer()
    # Specifies the path to temporarily save the uploaded audio file
    audio_path = "temp_audio.wav"
    # Saves the uploaded audio file to the specified path
    audio_file.save(audio_path)

    # Extracts features from the saved audio
    features = extract_features(audio_path)
    # Uses the pre-trained model to predict the probability that the speaker is the target speaker
    speaker_prob = model.predict_proba([features])[0][1]

    # Adjusted threshold for recognizing the target speaker
    if speaker_prob < 0.7:  # Change this value to be more strict or lenient
        return jsonify({'error': 'Speaker not recognized'})

    # Opens the audio file for reading
    with sr.AudioFile(audio_path) as source:
        # Records the audio from the file
        audio = recognizer.record(source)
        try:
            # Uses the 'Google Web Speech API' to recognize speech in the audio
            text = recognizer.recognize_google(audio, language="tr-TR")
            # Returns the recognized text as a JSON response
            return jsonify({'text': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Speech was unintelligible'})
        except sr.RequestError:
            return jsonify({'error': 'Could not request results from Google Speech Recognition service'})

if __name__ == '__main__':
    app.run(debug=True)
