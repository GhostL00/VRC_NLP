# Voice Translator — Streamlit App (Enhanced)
# File: app.py
# Purpose: Record or upload voice(s) -> Speech-to-Text -> Translate -> Text-to-Speech -> Playback
# Features added in this enhanced version:
#  - Human-readable language selection dropdown (language name + code)
#  - Batch translation: upload multiple audio files (zip or multiple file uploader)
#  - Real-time "live" streaming translation mode (short-chunk continuous listening -> translate -> speak)
#  - Offline TTS option using pyttsx3 (no external API) and online gTTS fallback
#  - Optional Whisper local STT support
#  - All free tools/options; user may choose online or local components
#
# Important setup notes (summary):
#  - Install Python packages: pip install -r requirements.txt
#  - requirements.txt should include: streamlit, speechrecognition, pydub, deep-translator, gTTS, langdetect, pyttsx3
#    Optional: whisper (pip install -U openai-whisper) but models are large
#  - Install ffmpeg (required by pydub for playback)
#  - PyAudio is required for microphone recording (on Windows use wheel from Gohlke if pip fails)
#
# Run: streamlit run app.py

import streamlit as st
import os
import tempfile
import time
import threading
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import zipfile
import io

# Offline TTS
try:
    import pyttsx3
    has_pyttsx3 = True
except Exception:
    has_pyttsx3 = False

# Optional Whisper
try:
    import whisper
    has_whisper = True
except Exception:
    has_whisper = False

DetectorFactory.seed = 0

st.set_page_config(page_title="Voice Translator — Enhanced", layout="wide")
st.title("🎙️ Voice Translator — Enhanced")

st.markdown("""
Enhanced features:
- Human-readable language dropdown
- Batch translation of multiple uploaded audio files (zip or multiple files)
- Real-time streaming translation mode (continuous short recordings)
- Offline TTS option using `pyttsx3` (no internet required for TTS)
""")

# ====== Language mapping (human-readable names) ======
LANGUAGES = {
    'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr', 'Spanish': 'es', 'French': 'fr',
    'German': 'de', 'Chinese (Simplified)': 'zh-cn', 'Japanese': 'ja', 'Russian': 'ru',
    'Arabic': 'ar', 'Portuguese': 'pt', 'Bengali': 'bn', 'Urdu': 'ur', 'Punjabi': 'pa'
}

# Reverse mapping for display
LANG_OPTIONS = [f"{name} — {code}" for name, code in LANGUAGES.items()]

# ====== Sidebar settings ======
st.sidebar.header("Settings")
stt_backend = st.sidebar.selectbox("Speech-to-Text backend", [
    "Google Web Speech (online)",
    "Whisper (local, optional)"
])

translate_choice = st.sidebar.selectbox("Target language", LANG_OPTIONS, index=1)
translate_target = translate_choice.split('—')[-1].strip()

tts_choice = st.sidebar.selectbox("Text-to-Speech (TTS)", ["gTTS (online)", "pyttsx3 (offline)"])

use_lang_detect = st.sidebar.checkbox("Auto-detect source language", value=True)
save_output = st.sidebar.checkbox("Save generated audio files to working directory", value=True)

# Real-time streaming params
st.sidebar.markdown("---")
st.sidebar.subheader("Live translation settings")
live_chunk_seconds = st.sidebar.slider("Chunk length (sec)", 1, 8, 4)
live_pause_between = st.sidebar.slider("Pause between chunks (sec)", 0, 3, 0)

# ====== Helpers ======

def save_uploaded_to_tempfile(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return tf.name


def recognize_with_google(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.warning(f"Google STT error: {e}")
        return ""


def recognize_with_whisper(audio_path, model_name='small'):
    if not has_whisper:
        st.warning("Whisper not installed. Install 'whisper' package to use local Whisper.")
        return ""
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        return result.get('text','')
    except Exception as e:
        st.warning(f"Whisper error: {e}")
        return ""


def translate_text(text, target_lang):
    if not text:
        return ""
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation error: {e}")
        return ""


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return None


def tts_gtts_and_save(text, lang, out_path):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(out_path)
        return out_path
    except Exception as e:
        st.warning(f"gTTS error: {e}")
        return None


def tts_pyttsx3_and_save(text, out_path):
    if not has_pyttsx3:
        st.warning("pyttsx3 not installed. Install with `pip install pyttsx3` to use offline TTS.")
        return None
    try:
        engine = pyttsx3.init()
        # Some engines allow setting voice/language — user may configure locally
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path
    except Exception as e:
        st.warning(f"pyttsx3 error: {e}")
        return None


def play_audio_file(path):
    try:
        audio = AudioSegment.from_file(path)
        play(audio)
    except Exception as e:
        st.warning(f"Audio playback error: {e}. File saved at {path}")


# ====== UI: Input selection ======
col1, col2 = st.columns([1,1])
with col1:
    st.header("Input")
    input_mode = st.radio("Choose input method", ["Record from microphone", "Upload audio file(s)", "Type text manually (skip STT)"])

with col2:
    st.header("Modes")
    live_mode = st.checkbox("Live translation streaming (continuous)", value=False)
    batch_mode = st.checkbox("Batch translation (multiple files)", value=False)

uploaded_paths = []
recorded_tempfile = None

if input_mode == "Upload audio file(s)":
    if batch_mode:
        uploads = st.file_uploader("Upload multiple audio files (or a zip) ", type=["wav","mp3","m4a","ogg","flac","zip"], accept_multiple_files=True)
        if uploads:
            # handle if a zip is uploaded among files
            for up in uploads:
                if up.type == 'application/zip' or up.name.endswith('.zip'):
                    # extract zip in memory
                    with zipfile.ZipFile(io.BytesIO(up.getbuffer())) as z:
                        for fname in z.namelist():
                            if fname.lower().endswith(('.wav','.mp3','.m4a','.ogg','.flac')):
                                data = z.read(fname)
                                tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1])
                                tf.write(data)
                                tf.flush(); tf.close()
                                uploaded_paths.append(tf.name)
                else:
                    path = save_uploaded_to_tempfile(up)
                    uploaded_paths.append(path)
            st.success(f"Saved {len(uploaded_paths)} uploaded files")
    else:
        upload = st.file_uploader("Upload a WAV/MP3 file", type=["wav","mp3","m4a","ogg","flac"])
        if upload is not None:
            uploaded_paths = [save_uploaded_to_tempfile(upload)]
            st.success(f"Saved uploaded file")
            st.audio(uploaded_paths[0])

elif input_mode == "Record from microphone":
    st.info("Recording uses your microphone via PyAudio. If you have trouble, record externally and upload the file.")
    dur = st.slider("Max recording (seconds)", min_value=1, max_value=30, value=6)
    if st.button("Start recording"):
        with st.spinner("Recording..."):
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.6)
                    audio = r.listen(source, timeout=5, phrase_time_limit=dur)
                tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                with open(tf.name, 'wb') as f:
                    f.write(audio.get_wav_data())
                recorded_tempfile = tf.name
                st.success(f"Recorded audio saved")
                st.audio(recorded_tempfile)
            except Exception as e:
                st.error(f"Recording failed: {e}")

else:  # Manual text
    typed_text = st.text_area("Type text to translate")

# ====== Processing functions for batch and single ======

def process_single_file(audio_file, target_lang, tts_engine_choice):
    # STT
    if stt_backend.startswith("Google"):
        src_text = recognize_with_google(audio_file)
    else:
        src_text = recognize_with_whisper(audio_file)

    if not src_text:
        return {'error': 'STT failed or empty', 'source_text': '', 'translated': ''}

    detected = None
    if use_lang_detect:
        detected = detect_language(src_text)

    translated = translate_text(src_text, target_lang)

    # TTS
    filename = f"translated_{int(time.time())}.mp3"
    out_path = os.path.join(tempfile.gettempdir(), filename)
    if tts_engine_choice == 'gTTS (online)':
        tts_file = tts_gtts_and_save(translated, target_lang, out_path)
    else:
        # pyttsx3 offline; save as WAV then convert to mp3 via pydub if desired
        wav_path = out_path.replace('.mp3', '.wav')
        tts_file = tts_pyttsx3_and_save(translated, wav_path)
        if tts_file:
            # convert to mp3
            try:
                aud = AudioSegment.from_file(wav_path)
                aud.export(out_path, format='mp3')
                tts_file = out_path
            except Exception:
                pass

    return {
        'error': None,
        'source_text': src_text,
        'detected_lang': detected,
        'translated': translated,
        'tts_file': tts_file
    }

# Batch processing
if st.button("Run Translation Pipeline") and not live_mode:
    results = []
    inputs = []
    if input_mode == "Type text manually (skip STT)":
        if not typed_text:
            st.error("Type text or provide audio.")
        else:
            # translate typed text directly
            translated = translate_text(typed_text, translate_target)
            st.subheader("Translated")
            st.write(translated)
            # TTS
            filename = f"translated_{int(time.time())}.mp3"
            out_path = os.path.join(tempfile.gettempdir(), filename)
            if tts_choice.startswith('gTTS'):
                tts_file = tts_gtts_and_save(translated, translate_target, out_path)
            else:
                tts_file = tts_pyttsx3_and_save(translated, out_path.replace('.mp3','.wav'))
            if tts_file:
                st.audio(tts_file)
                if save_output:
                    try:
                        os.replace(tts_file, os.path.join(os.getcwd(), os.path.basename(tts_file)))
                        st.success("Saved TTS file to working dir")
                    except Exception:
                        pass
    else:
        # collect audio inputs
        if recorded_tempfile:
            inputs = [recorded_tempfile]
        elif uploaded_paths:
            inputs = uploaded_paths
        else:
            st.error("No audio inputs provided.")

        if inputs:
            with st.spinner("Processing files..."):
                for audio_path in inputs:
                    res = process_single_file(audio_path, translate_target, tts_choice)
                    results.append((audio_path, res))

            for idx, (path, res) in enumerate(results):
                st.markdown(f"---
**File #{idx+1}:** {os.path.basename(path)}")
                if res.get('error'):
                    st.error(res['error'])
                else:
                    st.write("Detected language:", res.get('detected_lang'))
                    st.write("Source text:")
                    st.write(res.get('source_text'))
                    st.write("Translated:")
                    st.write(res.get('translated'))
                    if res.get('tts_file'):
                        st.audio(res.get('tts_file'))
                        if save_output:
                            try:
                                dest = os.path.join(os.getcwd(), os.path.basename(res.get('tts_file')))
                                os.replace(res.get('tts_file'), dest)
                                st.success(f"Saved TTS to {dest}")
                            except Exception:
                                pass

# ====== Live translation streaming ======

_live_thread = None
_live_flag = {'run': False}

def live_loop(target_lang, chunk_sec, pause_sec, tts_engine_choice):
    r = sr.Recognizer()
    mic = None
    try:
        mic = sr.Microphone()
    except Exception as e:
        st.error(f"Microphone not available: {e}")
        _live_flag['run'] = False
        return

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.6)
        while _live_flag['run']:
            try:
                audio = r.record(source, duration=chunk_sec)
                tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                with open(tf.name, 'wb') as f:
                    f.write(audio.get_wav_data())
                # STT
                if stt_backend.startswith('Google'):
                    src_text = recognize_with_google(tf.name)
                else:
                    src_text = recognize_with_whisper(tf.name)

                if not src_text:
                    # nothing recognized; continue
                    time.sleep(pause_sec)
                    continue

                detected = None
                if use_lang_detect:
                    detected = detect_language(src_text)

                translated = translate_text(src_text, target_lang)

                # show transient info in Streamlit (use websocket updates)
                st.write(f"**Live Detected ({detected}):** {src_text}")
                st.write(f"**Live Translated ({target_lang}):** {translated}")

                # TTS and play
                fname = f"live_{int(time.time())}.mp3"
                out_path = os.path.join(tempfile.gettempdir(), fname)
                if tts_engine_choice.startswith('gTTS'):
                    tts_file = tts_gtts_and_save(translated, target_lang, out_path)
                else:
                    tts_file = tts_pyttsx3_and_save(translated, out_path.replace('.mp3','.wav'))
                    if tts_file:
                        try:
                            aud = AudioSegment.from_file(tts_file)
                            aud.export(out_path, format='mp3')
                            tts_file = out_path
                        except Exception:
                            pass

                if tts_file:
                    # play in background thread
                    threading.Thread(target=play_audio_file, args=(tts_file,), daemon=True).start()

                time.sleep(pause_sec)
            except Exception as e:
                st.warning(f"Live loop error: {e}")
                time.sleep(1)

# Start / stop controls for live
if live_mode:
    st.markdown("---")
    st.subheader("Live translation streaming")
    if st.button("Start live translation") and not _live_flag['run']:
        _live_flag['run'] = True
        _live_thread = threading.Thread(target=live_loop, args=(translate_target, live_chunk_seconds, live_pause_between, tts_choice), daemon=True)
        _live_thread.start()
        st.success("Live translation started — speak into your microphone.")
    if st.button("Stop live translation") and _live_flag['run']:
        _live_flag['run'] = False
        st.success("Stopping live translation — please wait a moment.")

# Footer
st.markdown("---")
st.markdown("**Notes & Troubleshooting**")
st.markdown(
    """
- For reliable microphone recording, ensure PyAudio is installed and your microphone is accessible.
- Live streaming uses short fixed-duration recordings. Lower chunk size reduces latency but increases API calls.
- Offline Whisper + pyttsx3 provides a near-offline pipeline but requires installing models and may be slower.
- If gTTS or Google STT rate-limits, consider installing local models (Whisper for STT; Coqui or pyttsx3 for TTS).
"""
)

st.caption("Built with free tools: SpeechRecognition (Google), deep-translator, gTTS, pydub. Optional: Whisper for local STT, pyttsx3 for offline TTS.")
