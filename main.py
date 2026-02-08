from flask import Flask, render_template, request
import speech_recognition as sr
from pydub import AudioSegment
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import pipeline

# Initialize Flask
app = Flask(__name__)

# ------------------ Hugging Face Pipelines ------------------
# TEXT summarizers
summarizer_t5 = pipeline("summarization", model="t5-small")
summarizer_flan_t5 = pipeline("summarization", model="google/flan-t5-small")

# AUDIO summarizers (after STT)
summarizer_distilbart = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer_flan_t5_base = pipeline("summarization", model="google/flan-t5-base")

# Sentiment Analyzer
sentiment_analyzer = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")

# ------------------ Speech to Text ------------------
def speech_to_text(audio_file):
    temp_path = "temp_audio.wav"
    audio = AudioSegment.from_file(audio_file)
    audio.export(temp_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    os.remove(temp_path)
    return text

# ------------------ Picking Best Summary ------------------
def pick_best_summary(text, summarizer_a, summarizer_b):
    # Model A
    output_a = summarizer_a(
        text if "summarize:" not in text else text,
        max_length=150,
        min_length=50,
        do_sample=False
    )[0]['summary_text']

    # Model B
    output_b = summarizer_b(
        text if "summarize:" not in text else text,
        max_length=150,
        min_length=50,
        do_sample=False
    )[0]['summary_text']

    # Decide which summary to keep (length heuristic for simplicity)
    if len(output_a) < len(output_b):
        final_summary = output_a
    else:
        final_summary = output_b

    # analysis (only first 512 chars for speed)
    sentiment = sentiment_analyzer(text[:512])[0]

    return final_summary, sentiment

# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/process", methods=["POST"])
def process():
    if "audio" in request.files and request.files["audio"].filename != "":
        # AUDIO input
        audio_file = request.files["audio"]
        text = speech_to_text(audio_file)

        if not text.strip():
            return render_template("chat.html", summary="⚠️ Could not recognize audio.")

        # Audio Summarizers (DistilBART + Flan-T5-base)
        summary, sentiment = pick_best_summary(text, summarizer_distilbart, summarizer_flan_t5_base)

    else:
        # TEXT input
        text = request.form.get("text")
        if not text or text.strip() == "":
            return render_template("chat.html", summary="⚠️ No input provided.")

        # Use Text Summarizers (T5-small + Flan-T5-small)
        summary, sentiment = pick_best_summary(text, summarizer_t5, summarizer_flan_t5)

    return render_template(
        "chat.html",
        original=text,
        summary=summary,
        sentiment=f"Tone: {sentiment['label']} (Confidence: {round(sentiment['score']*100,2)}%)"
    )

if __name__ == "__main__":
    app.run(debug=True)
