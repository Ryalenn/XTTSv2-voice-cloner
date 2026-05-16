# Voice Cloner XTTS v2

Local zero-shot voice cloning application built with XTTS v2 and Streamlit.

Upload or record voice samples, enter a sentence, choose a language, and generate multiple speech variations locally.

---

## Features

* zero-shot voice cloning with XTTS v2
* upload audio references
* record voice samples directly from microphone
* multilingual speech generation
* generation of 3 different variations
* automatic audio preprocessing
* local Streamlit interface
* downloadable `.wav` outputs

---

## How it works

The application sends:

* reference voice samples
* a text prompt
* a generation language

to XTTS v2.

The model analyzes the provided voice references and generates new speech with a similar vocal timbre.

To avoid relying on a single inference result, the app generates three slightly different versions using different sampling parameters.

---

## Audio preprocessing

Before inference, each reference sample is automatically:

* converted to mono
* resampled to 24 kHz
* volume normalized
* trimmed to remove silence

This helps improve generation consistency and stability.

---

## Limitations

XTTS v2 can produce convincing results, but voice cloning is not perfect.

The model may lose:

* accent details
* pronunciation habits
* subtle intonations
* emotional nuances

Results depend heavily on:

* reference audio quality
* microphone quality
* background noise
* selected language
* text formatting

Proper nouns, acronyms, numbers, or foreign words may require phonetic spelling for more natural pronunciation.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voice-cloner.git
cd voice-cloner
```

---

## First setup

Simply double-click:

```text
install.bat
```

This script will automatically:

* create a virtual environment
* install all dependencies
* prepare the project for launch

---

## Launch the application

After installation, simply double-click:

```text
launch.bat
```

The Streamlit interface will open automatically in your browser.

---

## install.bat

```bat
@echo off
cd /d "%~dp0"

python -m venv .venv
call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

pause
```

---

## launch.bat

```bat
@echo off
cd /d "%~dp0"

call .venv\Scripts\activate
streamlit run app.py

pause
```

---

## Requirements

Recommended:

* Python 3.10
* NVIDIA GPU with CUDA support
* 8 GB+ VRAM recommended

The application also works on CPU, but inference will be slower.

---

## Project structure

```text
voice-cloner/
│
├── app.py
├── requirements.txt
├── install.bat
├── launch.bat
└── README.md
```

---


## Disclaimer

This project is intended for educational and research purposes only.

Do not use voice cloning technology to impersonate someone without their consent.
