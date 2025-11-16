#!/usr/bin/env python3
"""
Gradio app for speech recognition using:
 - Hugging Face transformers pipeline (AutoModelForSpeechSeq2Seq + AutoProcessor)
 - optional faster-whisper (if installed)

Features:
 - Upload an audio file
 - Choose backend: "huggingface" or "faster-whisper" (if available)
 - Enable word timestamps (if supported by backend/model)
 - Force Hindi decoding tokens (keeps behavior from original snippet)
 - Chunk length control
"""

import os
import json
import tempfile
import traceback
from typing import Dict, Any, List

import gradio as gr
import torch

# Transformers imports (for HF pipeline)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    GenerationConfig,
    pipeline,
)

# Try to import faster-whisper (optional)
try:
    from faster_whisper import WhisperModel as FWWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FWWhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

# ----------------------
# Configuration defaults
# ----------------------
HF_MODEL_ID_DEFAULT = "vasista22/whisper-hindi-small"  # your model
HF_GEN_CFG_REF = "openai/whisper-small"  # generation config template to load

# --------------
# Helper helpers
# --------------
def get_device_index():
    """Return device index for HF pipeline: 0 for cuda:0, -1 for cpu"""
    return 0 if torch.cuda.is_available() else -1

def safe_save_uploaded_file(uploaded):
    """Gradio gives a temp file object; return a path on disk (string)."""
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return uploaded
    # uploaded is a temporary file-like object (dict) or file object
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path

# ------------------------
# Hugging Face backend
# ------------------------
class HFASR:
    def __init__(self, model_id=HF_MODEL_ID_DEFAULT, device_index=None):
        self.model_id = model_id
        self.device_index = device_index if device_index is not None else get_device_index()
        self.asr_pipeline = None
        self._load()

    def _load(self):
        # Load model and processor
        print(f"[HF] Loading model / processor: {self.model_id} on device {self.device_index}")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # ensure generation_config (helps with timestamps)
        try:
            base_gen_cfg = GenerationConfig.from_pretrained(HF_GEN_CFG_REF)
            self.model.generation_config = base_gen_cfg
        except Exception as e:
            print("[HF] Could not load reference GenerationConfig:", e)

        device = self.device_index
        # Build pipeline; chunk_length can be adjusted at inference time by passing kwargs to pipeline() or stored pipeline attr
        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=None,  # set at call time
            device=device,
        )

    def transcribe(self, audio_path: str, return_timestamps: str = None, force_hindi: bool = True, chunk_length_s: int = 30):
        # update pipeline chunk length if desired (recreate pipeline if needed)
        if self.asr_pipeline and getattr(self.asr_pipeline, "chunk_length_s", None) != chunk_length_s:
            # recreate pipeline with new chunk length
            self.asr_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=chunk_length_s,
                return_timestamps=None,
                device=self.device_index,
            )

        # Optionally set forced decoder ids for Hindi
        if force_hindi:
            try:
                forced = self.asr_pipeline.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
                # pipelines expect model.config.forced_decoder_ids
                self.asr_pipeline.model.config.forced_decoder_ids = forced
            except Exception as e:
                print("[HF] Could not set forced_decoder_ids:", e)

        # Call pipeline
        call_kwargs = {}
        if return_timestamps:
            call_kwargs["return_timestamps"] = return_timestamps

        print("[HF] Running pipeline ... (this may take a while)")
        result = self.asr_pipeline(audio_path, **call_kwargs)
        # result is usually dict with "text" and possibly "chunks"/"segments"/"word_timestamps"
        return result

# ------------------------
# faster-whisper backend
# ------------------------
class FWASR:
    def __init__(self, model_id="small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8"):
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper is not installed")
        self.device = device
        # Use int8 for CPU, float16 for CUDA
        if device == "cpu":
            self.compute_type = "int8"
        else:
            self.compute_type = compute_type
        self.model_id = model_id
        print(f"[FW] Loading faster-whisper model {model_id} on {device} compute_type={self.compute_type}")
        # NOTE: faster-whisper expects model size names like "tiny", "base", "small", "medium", "large"
        self.model = FWWhisperModel(self.model_id, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: str, language: str = "hi", task: str = "transcribe", beam_size: int = 5, vad_filter: bool = False, word_timestamps: bool = False):
        """
        Returns a dict with text and segments list (each with start, end, text).
        The exact object returned by faster-whisper is a tuple (segments, info),
        so we wrap it into a more consistent dict format matching HuggingFace output.
        """
        segments_generator, info = self.model.transcribe(
            audio_path, 
            beam_size=beam_size, 
            language=language, 
            vad_filter=vad_filter, 
            task=task,
            word_timestamps=word_timestamps
        )
        
        # Convert generator to list and process
        segments_list = list(segments_generator)
        text = "".join([s.text for s in segments_list])
        
        # Build chunks in HF format (word-level)
        chunks = []
        if word_timestamps:
            for s in segments_list:
                if hasattr(s, 'words') and s.words:
                    for w in s.words:
                        chunks.append({
                            "text": w.word,
                            "timestamp": [float(w.start), float(w.end)]
                        })
        
        return {"text": text.strip(), "chunks": chunks}

# ------------------------
# Gradio app glue
# ------------------------
# Create singletons for HF and optional FW
HF_SINGLETON = None
FW_SINGLETON = None

def ensure_hf_loaded(model_id):
    global HF_SINGLETON
    if HF_SINGLETON is None or HF_SINGLETON.model_id != model_id:
        HF_SINGLETON = HFASR(model_id=model_id)
    return HF_SINGLETON

def ensure_fw_loaded(model_id):
    global FW_SINGLETON
    # Strip whitespace and convert common variations
    model_id = model_id.strip()
    if FW_SINGLETON is None or FW_SINGLETON.model_id != model_id:
        FW_SINGLETON = FWASR(model_id=model_id)
    return FW_SINGLETON

def generate_player_html(audio_path, chunks):
    """
    Generate interactive audio player with word highlighting.
    """
    if not chunks:
        return "<p>No timestamps available for highlighting</p>"
    
    # Generate unique ID for this instance
    unique_id = abs(hash(audio_path)) % 10000
    
    # Build the transcript HTML with inline onclick handlers
    transcript_html = ""
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        timestamp = chunk.get("timestamp", [0, 0])
        start_time = timestamp[0] if len(timestamp) > 0 else 0
        end_time = timestamp[1] if len(timestamp) > 1 else 0
        
        transcript_html += f'<span class="word" data-start="{start_time}" data-end="{end_time}" data-index="{i}" onclick="document.getElementById(\'audio_{unique_id}\').currentTime={start_time}; document.getElementById(\'audio_{unique_id}\').play();">{text}</span>'
    
    html = f"""
    <style>
        .audio-player-container {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}

        .audio-player-inner {{
            background: white;
            border-radius: 12px;
            padding: 20px;
        }}

        .audio-player {{
            width: 100%;
            margin-bottom: 20px;
            border-radius: 8px;
            outline: none;
        }}

        .transcript-container {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            min-height: 150px;
            max-height: 400px;
            overflow-y: auto;
        }}

        .transcript-text {{
            line-height: 2.2;
            font-size: 18px;
            color: #212529;
        }}

        .word {{
            display: inline-block;
            padding: 4px 6px;
            margin: 2px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .word:hover {{
            background-color: #e3f2fd;
            transform: translateY(-2px);
        }}

        .word.highlighted {{
            background-color: #ffeb3b;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(255, 235, 59, 0.5);
            transform: scale(1.05);
            animation: pulse 0.5s ease-in-out;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.8; }}
        }}

        .player-header {{
            text-align: center;
            margin-bottom: 15px;
            color: white;
            font-size: 1.3em;
            font-weight: 600;
        }}
    </style>
    
    <div class="audio-player-container">
        <div class="player-header">🎵 Interactive Audio Player</div>
        <div class="audio-player-inner">
            <audio id="audio_{unique_id}" class="audio-player" controls 
                   ontimeupdate="(function(){{
                       const audio = this;
                       const currentTime = audio.currentTime;
                       const words = audio.parentElement.querySelectorAll('.word');
                       words.forEach(function(word) {{
                           const start = parseFloat(word.getAttribute('data-start'));
                           const end = parseFloat(word.getAttribute('data-end'));
                           if (currentTime >= start && currentTime <= end) {{
                               word.classList.add('highlighted');
                               word.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                           }} else {{
                               word.classList.remove('highlighted');
                           }}
                       }});
                   }}).call(this)">
                <source src="/gradio_api/file={audio_path}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>

            <div class="transcript-container">
                <div class="transcript-text">
                    {transcript_html}
                </div>
            </div>
        </div>
    </div>
    """
    
    return html

def transcribe_gradio(audio_file, backend, hf_model_id, fw_model_id, want_word_timestamps, force_hindi, chunk_length_s):
    """
    Main function wired to Gradio interface.
    """
    if audio_file is None:
        return "No audio uploaded", {}, "No audio", ""
    # gradio provides a temporary file-like object, save to path
    input_path = safe_save_uploaded_file(audio_file)

    try:
        if backend == "huggingface":
            hf = ensure_hf_loaded(hf_model_id)
            rt_arg = "word" if want_word_timestamps else None
            res = hf.transcribe(input_path, return_timestamps=rt_arg, force_hindi=force_hindi, chunk_length_s=chunk_length_s)
            # Normalize result into text + segments if possible
            text = res.get("text", "")
            
            # Extract timestamps/chunks
            segments = []
            if "chunks" in res and res["chunks"]:
                # Word-level or chunk-level timestamps
                segments = res["chunks"]
            elif "segments" in res and res["segments"]:
                segments = res["segments"]
            
            # Generate player HTML
            player_html = generate_player_html(input_path, segments)
            
            return text, segments, json.dumps(res, default=str, ensure_ascii=False, indent=2), player_html

        elif backend == "faster-whisper":
            if not FASTER_WHISPER_AVAILABLE:
                return "faster-whisper not installed on server", {}, "faster-whisper not available", ""
            fw = ensure_fw_loaded(fw_model_id)
            res = fw.transcribe(input_path, language="hi", word_timestamps=want_word_timestamps)
            text = res.get("text", "")
            segments = res.get("chunks", [])
            
            # Generate player HTML
            player_html = generate_player_html(input_path, segments)
            
            return text, segments, json.dumps(res, default=str, ensure_ascii=False, indent=2), player_html

        else:
            return f"Unknown backend: {backend}", {}, "", ""

    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR]", tb)
        return f"Error during transcription: {e}\n\n{tb}", {}, "", ""

# ------------------------
# Build Gradio UI
# ------------------------
with gr.Blocks(title="Bhasha ASR demo (HF + optional faster-whisper)") as demo:
    gr.Markdown("# Bhasha ASR demo — Hugging Face / faster-whisper")
    gr.Markdown("""
    **Instructions:**
    - **Hugging Face backend**: Use full model IDs like `vasista22/whisper-hindi-small`
    - **faster-whisper backend**: Use model sizes only: `tiny`, `base`, `small`, `medium`, or `large`
    """)
    with gr.Row():
        with gr.Column(scale=2):
            audio_in = gr.Audio(label="Upload audio", type="filepath")
            backend = gr.Radio(choices=["huggingface", "faster-whisper"], value="faster-whisper", label="Backend")
            hf_model_id = gr.Textbox(label="HF model id", value=HF_MODEL_ID_DEFAULT, interactive=True)
            fw_model_id = gr.Textbox(label="faster-whisper model size", value="small", placeholder="tiny, base, small, medium, or large")
            chunk_length_s = gr.Slider(minimum=5, maximum=60, step=1, value=30, label="Chunk length (s)")
            want_word_timestamps = gr.Checkbox(label="Request word-level timestamps (if model supports it)", value=True)
            force_hindi = gr.Checkbox(label="Force Hindi decoding tokens", value=True)
            run_btn = gr.Button("Transcribe")
        with gr.Column(scale=3):
            txt_out = gr.Textbox(label="Transcription (plain text)", lines=8)
            html_out = gr.HTML(label="🎵 Interactive Audio Player with Word Highlighting")
            seg_out = gr.JSON(label="Segments / chunks (if available)")
            raw_out = gr.Textbox(label="Raw result (JSON-ish)", lines=12)

    def _trigger(audio, backend_choice, hf_mid, fw_mid, want_ts, force_hi, chunk_len):
        return transcribe_gradio(audio, backend_choice, hf_mid, fw_mid, want_ts, force_hi, int(chunk_len))

    run_btn.click(_trigger, inputs=[audio_in, backend, hf_model_id, fw_model_id, want_word_timestamps, force_hindi, chunk_length_s], outputs=[txt_out, seg_out, raw_out, html_out])
 

# Run
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
