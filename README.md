# Akbank-ML-Project (VoiceGuard)

VoiceGuard is an application that listens for a wake word ("Hey Akbank") and records the subsequent speech. The speech is then processed to verify if it belongs to a target speaker. If the verification is successful, the speech is converted to text and sent to ChatGPT for a response. The response from ChatGPT is displayed on the web interface.

## Features
- Wake word detection ("Hey Akbank")
- Audio recording and processing
- Speaker verification
- Speech-to-text conversion
- Integration with OpenAI's ChatGPT for generating responses
- Responsive web interface with a chat-like UI

## Requirements
- Python 3.x
- Flask
- OpenAI API key
- Joblib
- SpeechRecognition
- Librosa
- Dotenv
- HTML, CSS, and JavaScript

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/berkin-oztekin/Akbank-ML-Project
    cd Akbank-ML-Project
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the project root directory and add your OpenAI API key:
        ```
        API_KEY=your_openai_api_key
        ```

## Usage

1. Train the speaker recognition model (if not already trained):
    - Ensure you have voice samples for the target speaker and other speakers in the `model_training/voice_samples/` directory.
    - Run the training script:
        ```sh
        python train_model.py
        ```

2. Run the Flask application:
    ```sh
    python app.py
    ```

3. Open your web browser and go to `http://127.0.0.1:5000`.

## Project Structure

    
    Akbank-ML-Project (VoiceGuard)/
    │
    ├── model_training/
    │ ├── voice_samples/
    │ │ ├── target_speaker/
    │ │ └── other_speakers/
    │ ├── train_model.py
    │ └── speaker_recognition_model.pkl
    │
    ├── static/
    │ ├── css/
    │ │ ├── index_styles.css
    │ │ └── hey_akbank_styles.css
    │ ├── js/
    │ │ └── recorder.js
    │ └── images/
    │ └── akbank.jpg
    │
    ├── templates/
    │ ├── index.html
    │ └── hey_akbank.html
    │
    ├── utils/
    │ └── feature_extraction.py
    │
    ├── .env
    ├── app.py
    ├── requirements.txt
    └── README.md
    

## Code Overview

### `app.py`
This is the main Flask application file that defines the routes and logic for processing audio files, verifying speakers, and interacting with ChatGPT.

### `index.html`
This file provides the main interface for uploading audio files and recording audio for speaker verification.

### `hey_akbank.html`
This file provides a specialized interface for detecting the wake word "Hey Akbank," recording the subsequent speech, and displaying ChatGPT responses.

### `feature_extraction.py`
This script contains the logic for extracting audio features using Librosa.

### `train_model.py`
This script is used to train the speaker recognition model using voice samples.

## Endpoints

### `/`
Renders the main page (`index.html`).

### `/hey_akbank`
Renders the specialized "Hey Akbank" page (`hey_akbank.html`).

### `/speech-to-text` (POST)
Processes an uploaded audio file, verifies the speaker, and converts speech to text.

### `/process-speech` (POST)
Processes an uploaded audio file, verifies the speaker, and converts speech to text.

### `/process-target-speech` (POST)
Processes an uploaded audio file and saves it as a voice sample for the target speaker.

### `/chatgpt` (POST)
Sends the text prompt to ChatGPT and returns the response.

## Notes
- Ensure your microphone is connected and enabled when using the application.
- Allow the browser to access your microphone when prompted.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements
- [OpenAI](https://www.openai.com) for providing the ChatGPT API.
- [Librosa](https://librosa.org) for audio feature extraction.
- [Flask](https://flask.palletsprojects.com) for the web framework.

## Contact
For any inquiries or support, please contact: 
- https://github.com/berkin-oztekin 
- https://github.com/aliilman
- https://github.com/akinmertbur
