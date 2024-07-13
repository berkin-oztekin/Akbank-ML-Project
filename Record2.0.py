import pyaudio as pyaud
import wave
import threading
import speech_recognition as sr


FORMAT = pyaudio.paInt16    #Ses Özelliklerini giriyoruz
CHANNELS = 2
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav" #yazdığımız dosya
SILENCE_THRESHOLD = 1.5 #Sesizlik lagılandıktan 1.5 saniye sonra konuşma sessionunu bitiriyor

audio = pyaudio.PyAudio()


recording = True

def record_audio():#Thread
    global recording

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK) #istenilen recording işlemi stream objesi yapıyor.
    print("Voice Recording...")
    frames = []

    while recording: #kayıt işlemi sessizlik anında  recorrding değeri false dönücek ve session bittecek
        data = stream.read(CHUNK)
        frames.append(data)

    print("Record Stopped.")
    #Session bittikten sonra stream kapatılacakve dosyaya yazılacak
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def detect_silence():#Thread
    global recording
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=RATE) as source:
        while recording:
            print("Listening...")
            audio_data = recognizer.listen(source, phrase_time_limit=SILENCE_THRESHOLD)    #Thread phrase_time_limit=SILENCE_THRESHOLD sayesinde sessizlikten sonra sessionu bitirebiliyoruz
            try:
                recognizer.recognize_google(audio_data, language='tr-TR')
            except sr.UnknownValueError:
                print("Silence Detected, record stopped...")
                recording = False    #recording objesinin false dönmesi kayıt işlemini sonlandırır
                break
            except sr.RequestError as e:
                print("Error occurred; {0}".format(e))
                recording = False
                break


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='tr-TR')
            print("Recognized text: " + text)
        except sr.UnknownValueError:
            print("Unrecognized Voice.")
        except sr.RequestError as e:
            print("Error Occurred; {0}".format(e))


record_thread = threading.Thread(target=record_audio)
silence_thread = threading.Thread(target=detect_silence)

record_thread.start()
silence_thread.start()

record_thread.join()
silence_thread.join()

# Kaydedilen sesi tanıma
recognize_speech()
print("Voice saved and already transformed the text.")
