# pip install -U transformers accelerate torch
# If you still want faster-whisper alternative, tell me.

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    GenerationConfig,
    pipeline,
)
import pprint

audio = "sample.mp3"
device = 0 if torch.cuda.is_available() else -1  # pipeline expects int device index or -1 for CPU

model_id = "vasista22/whisper-hindi-small"

# 1) load model + processor (processor contains tokenizer + feature_extractor)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# 2) Ensure the model has a proper generation_config (needed for return_timestamps)
#    Use a canonical OpenAI Whisper generation config that includes timestamp-related fields.
#    Choose the size that best matches your model (tiny/base/small/medium/large).
base_gen_cfg = GenerationConfig.from_pretrained("openai/whisper-small")
model.generation_config = base_gen_cfg

# 3) If your model repo contains forced_decoder_ids overrides, preserve them:
if hasattr(model.config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids

# 4) Build pipeline, explicitly supplying tokenizer and feature_extractor
asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    return_timestamps="word",   # request word-level timestamps (if supported)
    device=device
)

# 5) Optionally force Hindi decoding tokens (keeps behavior from your original snippet)
asr.model.config.forced_decoder_ids = asr.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

# 6) Run
result = asr(audio)

pprint.pprint(result)
# Typical keys to inspect: "text", "chunks" or "segments" or "word_timestamps"
