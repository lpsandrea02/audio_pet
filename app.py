import os
from flask import Flask, jsonify, render_template, request, send_from_directory
from audio_tools import (normalize_audio, detect_melody, synthesise_output,
                         determine_emotion, generate_click_noise,
                         generate_idle_noise, get_idle_noise_delay,
                         add_to_memory, short_term_memory)

app = Flask(__name__)

# Core directory configurations for file handling
UPLOAD_FOLDER = os.path.join("data", "recordings")
RESPONSE_FOLDER = os.path.join("data", "responses")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESPONSE_FOLDER, exist_ok=True)

# =====================================================================
# 1. AUDIO & DECISION LOGIC
# =====================================================================
def analyze_and_generate_reply(input_wav_path, character="default_cat"):
    """
    Core logic block. Process the received audio file
    and decide what response file, skin, and emotion to emit back.

    Runs the full Audiopet audio pipeline on a user recording:
    normalisation, melody detection, emotion decision, and response
    synthesis in the selected character's voice. Invalid or empty
    recordings are handled inside the audio_tools pipeline, which
    synthesises a preset confused noise in that case.

    Args:
        input_wav_path (str): Path to the uploaded user recording (.wav).
        character (str, optional): Character name matching a key in
            ``VOICE_PROFILES``. Defaults to "default_cat".

    Returns:
        dict: Payload for the frontend with keys:
            - "audio_url" (str): URL from which the reply .wav can be streamed.
            - "skin" (str): The active character name.
            - "emotion" (str): Emotion the reply was delivered in
              (e.g. "happy", "confused").
    """
    print(f"-> Flask processing payload saved at: {input_wav_path}")

    normalize_audio(input_wav_path, "data/recordings/boosted_input.wav")
    note_sequence = detect_melody("data/recordings/boosted_input.wav")

    # Remember this melody for later recall on character clicks.
    # add_to_memory() silently ignores empty logs, so confused-noise
    # fallbacks (no detected melody) never pollute the short-term memory.
    add_to_memory(note_sequence, memory=short_term_memory)

    output_audio_filename = "system_reply.wav"
    output_audio_path = os.path.join(RESPONSE_FOLDER, output_audio_filename)

    emotion = determine_emotion(note_sequence, input_wav_path)

    emotion = synthesise_output(note_sequence, 
                    output_filename=output_audio_path, 
                    sample_rate=44100,
                    character=character,
                    emotion=emotion) 
    
    return {
        "audio_url": f"/stream-audio/{output_audio_filename}", 
        "skin": character,
        "emotion": emotion
    }

# =====================================================================
# 2. FLASK ROUTING MATRIX
# =====================================================================
@app.route("/")
def index():
    """Render and serve the main HTML interface (templates/index.html).

    Returns:
        str: Rendered HTML of the Audiopet interaction interface.
    """
    return render_template("index.html")

@app.route("/api/process-audio", methods=["POST"])
def process_audio():
    """
    Unified route handling data uploads.
    Accepts browser multi-part forms containing audio binary data.

    Expects a multipart/form-data POST with:
        - "file": the recorded audio binary (.wav).
        - "current_skin" (optional): active character name; defaults
          to "default_cat" if blank.
        - "current_emotion" (optional): frontend-reported emotion;
          defaults to "happy" if blank.

    Returns:
        tuple: A ``(jsonify(payload), status_code)`` pair.
            On success (200), the payload is the decision dictionary
            from :func:`analyze_and_generate_reply`.
            On failure (400), the payload is ``{"error": "<message>"}``
            when the file payload or filename is missing.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file payload detected"}), 400
        
    audio_file = request.files["file"]
    
    # Extract the active character selection sent from the frontend screen
    # Defaults to 'default_cat' if the data field happens to be blank
    character_name = request.form.get("current_skin", "default_cat")
    current_emotion = request.form.get("current_emotion", "happy")

    if audio_file.filename == "":
        return jsonify({"error": "Empty filename property"}), 400

    # 1. Save incoming audio binary file to disk securely
    saved_input_path = os.path.join(UPLOAD_FOLDER, "incoming_recording.wav")
    audio_file.save(saved_input_path)

    # 2. Fire the custom decision tree analytics, passing the chosen character name
    decision_payload = analyze_and_generate_reply(saved_input_path, character=character_name)
    
    return jsonify(decision_payload)


@app.route("/api/character-click", methods=["POST"])
def character_click():
    """
    Handles pokes on the character sprite in the web app interface.

    Expects a POST form with:
        - "current_skin" (optional): active character name; defaults
          to "default_cat" if blank.
        - "click_count" (optional): how many times the user has poked
          the character; defaults to 0 if blank or unparseable.

    Synthesises the click noise into data/responses/system_noise.wav:
    occasionally (with MEMORY_SING_PROBABILITY) the Audiopet recalls a
    randomly chosen melody from its short-term memory and sings it with
    a happy or neutral delivery; otherwise it emits a happy noise or a
    neutral click noise (0.70 happy / 0.30 none), with a forced angry
    noise once the user clicks too many times. This
    file is deliberately separate from system_reply.wav, which stays
    reserved for direct user-to-audiopet interactions.

    Returns:
        Response: JSON payload with keys:
            - "audio_url" (str): URL of the click noise .wav.
            - "skin" (str): The active character name.
            - "emotion" (str): "happy", "none", or (once over-clicked)
              "angry".
    """
    character_name = request.form.get("current_skin", "default_cat")
    try:
        click_count = int(request.form.get("click_count", 0))
    except (TypeError, ValueError):
        click_count = 0

    noise_audio_filename = "system_noise.wav"
    noise_audio_path = os.path.join(RESPONSE_FOLDER, noise_audio_filename)

    emotion = generate_click_noise(
        character=character_name,
        click_count=click_count,
        output_filename=noise_audio_path,
        memory=short_term_memory
    )

    return jsonify({
        "audio_url": f"/stream-audio/{noise_audio_filename}",
        "skin": character_name,
        "emotion": emotion
    })


@app.route("/api/idle-noise", methods=["POST"])
def idle_noise():
    """
    Handles the character's idle noises in the web app interface.

    Expects a POST form with:
        - "current_skin" (optional): active character name; defaults
          to "default_cat" if blank.
        - "idle_count" (optional): how many idle noises have already
          been emitted in the current idle session; defaults to 0 if
          blank or unparseable.

    Synthesises the idle noise into data/responses/system_idle.wav:
    while the idle count is below IDLE_NOISE_MAX_COUNT the noise is one
    of the preset happy/confused/angry melodies (or occasionally a
    randomly chosen melody recalled from the short-term memory); once
    the count is reached the preset sad noise is delivered with the
    "sad" emotion instead and "is_final" is True — the frontend must
    stop scheduling idle noises after it. This file is deliberately
    separate from system_reply.wav (direct user interactions) and
    system_noise.wav (pokes).

    Returns:
        Response: JSON payload with keys:
            - "audio_url" (str): URL of the idle noise .wav.
            - "skin" (str): The active character name.
            - "emotion" (str): The idle noise emotion.
            - "is_final" (bool): True when the final sad noise played;
              no further idle noises may be requested.
    """
    character_name = request.form.get("current_skin", "default_cat")
    try:
        idle_count = int(request.form.get("idle_count", 0))
    except (TypeError, ValueError):
        idle_count = 0

    idle_audio_filename = "system_idle.wav"
    idle_audio_path = os.path.join(RESPONSE_FOLDER, idle_audio_filename)

    emotion, is_final = generate_idle_noise(
        character=character_name,
        idle_count=idle_count,
        output_filename=idle_audio_path,
        memory=short_term_memory
    )

    return jsonify({
        "audio_url": f"/stream-audio/{idle_audio_filename}",
        "skin": character_name,
        "emotion": emotion,
        "is_final": is_final,
        "next_delay": get_idle_noise_delay()
    })


@app.route("/api/idle-delay")
def idle_delay():
    """
    Provides the randomly spaced waiting time before the next idle
    character noise.

    The frontend schedules each idle noise after a random delay so
    consecutive noises never arrive at a fixed interval; this endpoint
    draws that delay from :func:`get_idle_noise_delay` (uniform between
    IDLE_NOISE_MIN_DELAY and IDLE_NOISE_MAX_DELAY seconds).

    Returns:
        Response: JSON payload with key "delay" (float seconds).
    """
    return jsonify({"delay": get_idle_noise_delay()})


@app.route("/stream-audio/<filename>")
def stream_audio(filename):
    """
    Serves the output wave files safely from the data storage directory.

    Args:
        filename (str): Name of the .wav file inside the responses
            folder to stream (e.g. "system_reply.wav").

    Returns:
        Response: The audio file stream with mimetype "audio/wav".
    """
    return send_from_directory(RESPONSE_FOLDER, filename, mimetype="audio/wav")


if __name__ == "__main__":
    # Start local development server on port 5000
    app.run(debug=True, port=5000)
