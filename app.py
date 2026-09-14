import os
from flask import Flask, jsonify, render_template, request, send_from_directory
from audio_tools import normalize_audio, detect_melody, synthesise_output

app = Flask(__name__)

# Core directory configurations for file handling
UPLOAD_FOLDER = os.path.join("data", "recordings")
RESPONSE_FOLDER = os.path.join("data", "responses")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESPONSE_FOLDER, exist_ok=True)

# =====================================================================
# 1. PLACEHOLDER: INSERT YOUR CUSTOM AUDIO & DECISION LOGIC HERE
# =====================================================================
def analyze_and_generate_reply(input_wav_path, character="default_cat"):
    """
    [YOUR HOOK] Core logic block. Process the received audio file 
    and decide what response file, skin, and emotion to emit back.
    """
    print(f"-> Flask processing payload saved at: {input_wav_path}")

    normalize_audio(input_wav_path, "data/recordings/boosted_input.wav")
    note_sequence = detect_melody("data/recordings/boosted_input.wav")

    output_audio_filename = "system_reply.wav"
    output_audio_path = os.path.join(RESPONSE_FOLDER, output_audio_filename)

    synthesise_output(note_sequence, 
                      output_filename=output_audio_path, 
                      sample_rate=44100,
                      character=character) 
    

    #output_audio_filename="cute_bubbly_happy.wav"

    return {
        "audio_url": f"/stream-audio/{output_audio_filename}", 
        "skin": character,
        "emotion": "happy"
    }

# =====================================================================
# 2. FLASK ROUTING MATRIX
# =====================================================================
@app.route("/")
def index():
    """Renders your main HTML interface file."""
    return render_template("index.html")

@app.route("/api/process-audio", methods=["POST"])
def process_audio():
    """
    Unified route handling data uploads. 
    Accepts browser multi-part forms containing audio binary data.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file payload detected"}), 400
        
    audio_file = request.files["file"]
    current_skin = request.form.get("current_skin", "default_cat")
    current_emotion = request.form.get("current_emotion", "happy")

    if audio_file.filename == "":
        return jsonify({"error": "Empty filename property"}), 400

    # 1. Save incoming audio binary file to disk
    saved_input_path = os.path.join(UPLOAD_FOLDER, "incoming_recording.wav")
    audio_file.save(saved_input_path)

    # 2. Fire the custom decision tree analytics
    decision_payload = analyze_and_generate_reply(saved_input_path, character="radio_robot")
    
    return jsonify(decision_payload)


@app.route("/stream-audio/<filename>")
def stream_audio(filename):
    """
    Serves the output wave files safely from the data storage directory.
    """
    return send_from_directory(RESPONSE_FOLDER, filename, mimetype="audio/wav")


if __name__ == "__main__":
    # Start local development server on port 5000
    app.run(debug=True, port=5000)
