import hashlib
import os
from pathlib import Path

import librosa
import soundfile as sf
import streamlit as st
import torch
from TTS.api import TTS


st.set_page_config(
    page_title="Voice Cloner",
    page_icon="🔊",
    layout="wide",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000;
            color: #ffffff;
            background-image:
                radial-gradient(circle at 20% 20%, rgba(255,255,255,0.08) 1px, transparent 1px),
                radial-gradient(circle at 80% 60%, rgba(255,255,255,0.06) 1px, transparent 1px);
            background-size: 8px 8px, 12px 12px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        header {
            background: transparent !important;
        }

        footer {
            visibility: hidden;
        }

        .portfolio-header {
            border: 1px solid #ffffff;
            padding: 1.5rem;
            margin-bottom: 2rem;
            background: rgba(0, 0, 0, 0.92);
        }

        .portfolio-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .portfolio-subtitle {
            font-size: 1rem;
            color: #d4d4d4;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 750;
            margin-bottom: 0.5rem;
        }

        .section-text {
            color: #cfcfcf;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .or-separator {
            text-align: center;
            font-size: 0.95rem;
            font-weight: 700;
            margin: 1rem 0;
            color: #ffffff;
        }

        .reference-count {
            color: #ffffff;
            font-size: 0.95rem;
            margin: 0.6rem 0 1.4rem;
        }

        .output-card {
            border: 1px solid #ffffff;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.92);
            margin-bottom: 1rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #ffffff !important;
            border-radius: 0 !important;
            background: rgba(0, 0, 0, 0.92) !important;
            padding: 1.25rem !important;
        }

        textarea, input {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
            border-radius: 0 !important;
        }

        [data-testid="stFileUploader"] {
            background: rgba(0, 0, 0, 0.92);
        }

        [data-testid="stFileUploader"] section {
            background-color: #000000 !important;
            border: 1px dashed #ffffff !important;
            border-radius: 0 !important;
        }

        [data-baseweb="select"] > div {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
            border-radius: 0 !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
            border-radius: 0 !important;
            font-weight: 700 !important;
            padding: 0.6rem 1rem !important;
        }

        audio {
            width: 100%;
            border-radius: 0;
        }

        label, .stMarkdown, p, span {
            color: #ffffff;
        }

        .small-muted {
            color: #a3a3a3;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

#state init
if "outputs" not in st.session_state:
    st.session_state.outputs = None

if "recordings" not in st.session_state:
    st.session_state.recordings = []

if "recording_hashes" not in st.session_state:
    st.session_state.recording_hashes = set()

#model setup
os.environ["COQUI_TOS_AGREED"] = "1"

BASE_DIR = Path.cwd()
UPLOAD_DIR = BASE_DIR / "uploads"
CLEAN_DIR = BASE_DIR / "clean_refs"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
CLEAN_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 10


@st.cache_resource
def load_model():
    return TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
    ).to(DEVICE)


tts = load_model()


XTTS_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}


#audio preprocessing
def preprocess_audio(input_path: Path, output_path: Path):
    try:
        audio, _ = librosa.load(input_path, sr=24000, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=30)

        if len(audio) == 0:
            return None

        max_abs = abs(audio).max()
        if max_abs > 0:
            audio = audio / max_abs * 0.9

        sf.write(output_path, audio, 24000)
        return str(output_path)

    except Exception as error:
        st.warning(f"Could not process {input_path.name}: {error}")
        return None


def save_reference_file(file, index: int):
    suffix = Path(file.name).suffix.lower() or ".wav"
    raw_path = UPLOAD_DIR / f"reference_{index}{suffix}"

    with open(raw_path, "wb") as f:
        f.write(file.getbuffer())

    return raw_path


def prepare_references(reference_files):
    clean_paths = []

    for i, file in enumerate(reference_files):
        raw_path = save_reference_file(file, i)
        clean_path = CLEAN_DIR / f"ref_{i}.wav"

        processed = preprocess_audio(raw_path, clean_path)

        if processed:
            clean_paths.append(processed)

    return clean_paths


def get_audio_hash(audio_file):
    return hashlib.md5(audio_file.getvalue()).hexdigest()


#generation
def generate_variants(text: str, refs: list[str], language: str):
    configs = [
        {"label": "Version 1", "temperature": 0.45, "top_p": 0.70, "top_k": 20},
        {"label": "Version 2", "temperature": 0.55, "top_p": 0.75, "top_k": 30},
        {"label": "Version 3", "temperature": 0.65, "top_p": 0.80, "top_k": 40},
    ]

    outputs = []

    for i, config in enumerate(configs):
        out_path = OUTPUT_DIR / f"variant_{i + 1}.wav"

        tts.tts_to_file(
            text=text,
            speaker_wav=refs,
            language=language,
            file_path=str(out_path),
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=5.0,
            length_penalty=1.0,
            speed=1.0,
            split_sentences=False,
        )

        outputs.append({"label": config["label"], "path": str(out_path)})

    return outputs


def display_recorded_samples():
    if not st.session_state.recordings:
        return

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">Recorded samples</div>
            <div class="section-text">
                Preview the recordings added to the reference list.
            <br><br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for recording in st.session_state.recordings:
            st.audio(recording.getvalue(), format="audio/wav")

        if st.button("Clear recordings"):
            st.session_state.recordings = []
            st.session_state.recording_hashes = set()
            st.session_state.outputs = None
            st.rerun()


#ui
st.markdown(
    """
    <div class="portfolio-header">
        <div class="portfolio-title">Voice Cloner</div>
        <div class="portfolio-subtitle">Upload or record voice samples, write a sentence, generate three versions.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        f"""
        <div class="section-title">Input</div>
        <div class="section-text">
            Add up to {MAX_SAMPLES} reference audio samples in total. Short, clean voice recordings work best.
            <br><br>
            <span class="small-muted">Device: {DEVICE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload reference audio files",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
    )

    st.markdown('<div class="or-separator" style="text-align:left; margin-top:1rem; margin-bottom:1rem;">Or</div>', unsafe_allow_html=True)

    recorded_audio = st.audio_input("Record a reference sample")

if recorded_audio is not None:
    total_count = len(uploaded_files or []) + len(st.session_state.recordings)
    recording_hash = get_audio_hash(recorded_audio)

    if recording_hash not in st.session_state.recording_hashes:
        if total_count >= MAX_SAMPLES:
            st.error(f"You can use up to {MAX_SAMPLES} reference audio samples in total.")
        else:
            st.session_state.recordings.append(recorded_audio)
            st.session_state.recording_hashes.add(recording_hash)
            st.session_state.outputs = None
            st.success("Recording added.")

display_recorded_samples()

total_samples = len(uploaded_files or []) + len(st.session_state.recordings)

st.markdown(
    f"""
    <div class="reference-count">
        {total_samples} / {MAX_SAMPLES} samples selected.
    </div>
    """,
    unsafe_allow_html=True,
)

language_name = st.selectbox(
    "Generation language",
    options=list(XTTS_LANGUAGES.keys()),
    index=0,
)

text = st.text_area(
    "Text to generate",
    value="Hello, this is a voice cloning test.",
    height=120,
)

generate_button = st.button("Generate")

if generate_button:
    all_reference_files = list(uploaded_files or []) + st.session_state.recordings

    if not all_reference_files:
        st.error("Please upload or record at least one audio sample.")
    elif len(all_reference_files) > MAX_SAMPLES:
        st.error(f"You can use up to {MAX_SAMPLES} reference audio samples in total.")
    elif not text.strip():
        st.error("Please enter some text.")
    else:
        language_code = XTTS_LANGUAGES[language_name]

        with st.spinner("Preparing reference audio..."):
            refs = prepare_references(all_reference_files)

        if len(refs) == 0:
            st.error("No valid audio files after preprocessing.")
            st.session_state.outputs = None
        else:
            with st.spinner("Generating audio..."):
                st.session_state.outputs = generate_variants(
                    text.strip(),
                    refs,
                    language_code,
                )

            st.success("Done.")


#display results
if st.session_state.outputs is not None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">Results</div>
            <div class="section-text">
                Listen to the three versions and download the one you prefer.
            </div>
            """,
            unsafe_allow_html=True,
        )

    cols = st.columns(3)

    for col, output in zip(cols, st.session_state.outputs):
        with col:
            st.markdown(
                f"""
                <div class="output-card">
                    <strong>{output["label"]}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with open(output["path"], "rb") as audio_file:
                audio_bytes = audio_file.read()

            st.audio(audio_bytes, format="audio/wav")

            st.download_button(
                label="Download",
                data=audio_bytes,
                file_name=Path(output["path"]).name,
                mime="audio/wav",
                key=f"download_{output['label']}",
            )