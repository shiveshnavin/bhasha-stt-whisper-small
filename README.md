---
title: Bhasha STT Whisper Small
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Bhasha STT Whisper Small

A speech-to-text (STT) application for Hindi audio transcription using Whisper models. This application supports both Hugging Face transformers and faster-whisper backends for flexible and efficient inference.

## Features

- 🎯 **Dual Backend Support**: Choose between Hugging Face transformers or faster-whisper for inference
- ⏱️ **Word-level Timestamps**: Get precise timing information for transcribed words
- 🇮🇳 **Hindi Language Support**: Optimized for Hindi speech recognition with forced decoder tokens
- 🔧 **Configurable Chunk Processing**: Adjust chunk length for optimal memory usage
- 🚀 **GPU Acceleration**: Automatic CUDA detection and support for faster inference

## Quick Start

### Using the Gradio Interface

1. Upload an audio file (WAV, MP3, etc.)
2. Select your preferred backend:
   - **Hugging Face**: Use full model IDs like `vasista22/whisper-hindi-small`
   - **faster-whisper**: Use model sizes: `tiny`, `base`, `small`, `medium`, or `large`
3. Configure options:
   - Enable word-level timestamps
   - Force Hindi decoding tokens
   - Adjust chunk length (5-60 seconds)
4. Click "Transcribe" to process

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Gradio
- Accelerate
- faster-whisper (optional, for faster-whisper backend)

## Usage

### Running the App

```bash
python app.py
```

The app will launch on `http://0.0.0.0:7860` by default.

### Backends

#### Hugging Face Backend

Uses the transformers library with `AutoModelForSpeechSeq2Seq`. Supports any Whisper-compatible model from Hugging Face Hub.

Default model: `vasista22/whisper-hindi-small`

#### faster-whisper Backend

Uses CTranslate2 for optimized inference. Requires faster-whisper to be installed.

Supported model sizes: `tiny`, `base`, `small`, `medium`, `large`

## Configuration

### Chunk Length

Adjust the chunk length parameter (5-60 seconds) to balance between:
- **Shorter chunks**: Lower memory usage, better for short audio
- **Longer chunks**: Better context, suitable for longer audio

### Word Timestamps

Enable word-level timestamps to get precise timing information for each transcribed word. Note that accuracy depends on the model's capabilities.

### Force Hindi Decoding

Forces the model to decode in Hindi by setting appropriate decoder prompt IDs. Recommended for Hindi-only audio.

## Model Information

This application is configured to use `vasista22/whisper-hindi-small` by default, a Whisper model fine-tuned for Hindi speech recognition.

## Performance Notes

- GPU inference is automatically enabled when CUDA is available
- Use `float16` precision for faster GPU inference
- Consider using faster-whisper backend for production deployments requiring low latency
- For frame-accurate word timestamps, consider post-processing with alignment tools like WhisperX

## License

Apache 2.0

## Citation

If you use this application, please cite the original Whisper paper:

```bibtex
@misc{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  year={2022},
  eprint={2212.04356},
  archivePrefix={arXiv}
}
```

## Acknowledgments

Built with:
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Gradio](https://github.com/gradio-app/gradio)
