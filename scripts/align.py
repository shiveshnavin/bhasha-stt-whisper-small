import whisperx
import json
import soundfile as sf

audio_file = "audio.wav"
device = "cpu"

with sf.SoundFile(audio_file) as f:
    duration = len(f) / f.samplerate   # seconds

audio = whisperx.load_audio(audio_file)

model_a, metadata = whisperx.load_align_model(language_code="hi", device=device)
with open("text.txt", "r", encoding="utf8") as f:
    transcript_text = f.read().strip()
formatted_transcript = [{
    "text": transcript_text,
    "start": 0.0,
    "end": round(duration, 2)
}]
result = whisperx.align(formatted_transcript, model_a, metadata, audio, device)

# Directly use result["word_segments"] instead of saving/loading word_timestamps.json
word_data = result["word_segments"]

# Hardcoded transcript (replace if needed)
transcript_text = ' '.join([w["word"] for w in word_data])

words_out = []
for i, w in enumerate(word_data):
    words_out.append({
        "alignedWord": w["word"],
        "start": w["start"],
        "end": w["end"],
        "word": "",          # placeholder for phonetic / latin form
        "idx": i,
        "case": "success",
        "phenomes": []
    })

output = {
    "model": "small",
    "transcript": transcript_text,
    "words": words_out,
    "start": word_data[0]["start"] if word_data else 0,
    "end": word_data[-1]["end"] if word_data else 0
}

# Save output
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("output.json generated.")
