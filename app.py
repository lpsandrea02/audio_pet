import os
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from audio_tools import (normalize_audio, detect_melody, synthesise_output,
                         determine_emotion, generate_click_noise,
                         generate_idle_noise, get_idle_noise_delay,
                         add_to_memory, short_term_memory,
                         is_user_turn_ended, should_interrupt_angry,
                         extend_melody_log, generate_interrupt_reply,
                         generate_interrupt_angry_noise, is_chunk_voiced,
                         _get_raw_audio_duration,
                          combine_melody_sections, replace_last_section,
                          generate_lesson_reply, add_to_lesson_memory,
                          pick_lesson_memory_melody, long_term_memory,
                          should_sing_from_memory,
                          make_mistake_melody, should_lesson_mistake,
                          pick_lesson_redo_noise, generate_lesson_noise,
                          LESSON_MEMORY_SING_PROBABILITY,
                         determine_noise_emotion, should_emit_idle_noise,
                         CLICK_NOISE_ANGER_THRESHOLD)

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
# 1.5 INTERACTIVE MODE SESSION STATE
# =====================================================================
# Full-duplex interactive session: the frontend continuously POSTs short
# audio chunks while the user is talking; this module-level state tracks
# the melody accumulated in the current user turn, how much silence has
# passed since the latest detected note, and how many times the user has
# interrupted the pet's reply (reset once the angry noise fires).
interactive_session = {
    "active": False,
    "melody_log": [],
    "silence_seconds": 0.0,
    "interrupt_count": 0,
}


def reset_interactive_session(active=False):
    """
    Resets (and optionally activates) the interactive-mode session.

    Clears the accumulated turn melody, the silence counter, and the
    interruption counter, setting the session active flag to the given
    value.

    Args:
        active (bool, optional): Whether the session should be marked
            active after the reset. Defaults to False.

    Returns:
        None.
    """
    interactive_session.update(
        active=active,
        melody_log=[],
        silence_seconds=0.0,
        interrupt_count=0,
    )


def process_interactive_chunk(chunk_wav_path, pet_speaking,
                              character="default_cat"):
    """
    Processes one streamed audio chunk for the interactive session.

    Runs the chunk through the gated detection pipeline: the chunk is
    normalised, then the WebRTC voice-activity gate
    (:func:`is_chunk_voiced`) decides whether it is note-analysed at
    all — unvoiced chunks (background noise, silence) skip melody
    detection entirely so they cleanly accumulate into the turn-end
    silence clock. The session state is then updated:

    - Notes detected while the user's turn is active are merged into
      the running turn melody and reset the silence clock.
    - Notes detected while the pet is speaking count as an
      interruption: the interrupt counter is incremented and the
      previous melody is discarded, so only the new note sequence is
      synthesised when the next turn ends.
    - A chunk without detected notes accumulates its decoded duration
      into the silence clock.

    When the measured silence meets INTERACTIVE_SILENCE_THRESHOLD and
    at least one note was detected in the turn, the turn ends: if the
    interrupt count has reached INTERACTIVE_INTERRUPT_ANGER_THRESHOLD
    the pet renders the forced angry noise (and resets the counter);
    otherwise it synthesises the accumulated melody in the character's
    voice and stores it in short-term memory. In both cases the turn
    melody and silence clock are cleared afterwards.

    Args:
        chunk_wav_path (str): Path to the saved chunk audio file.
        pet_speaking (bool): True when the Audiopet's reply audio is
            currently playing in the frontend.
        character (str, optional): Character name matching a key in
            ``VOICE_PROFILES``. Defaults to "default_cat".

    Returns:
        dict: Chunk result payload with keys:
            - "notes_detected" (int): Notes found in this chunk.
            - "total_notes" (int): Notes accumulated in the turn so far.
            - "interrupted" (bool): True if this chunk was an
              interruption of the pet's singing.
            - "interrupt_count" (int): Current interrupt counter.
            - "turn_ended" (bool): True when the turn ended and a
              reply was synthesised.
            - "audio_url" (str or None): Reply/noise stream URL when
              the turn ended, else None.
            - "emotion" (str or None): Reply emotion when the turn
              ended, else None.
    """
    boosted_path = os.path.join(UPLOAD_FOLDER,
                                "interactive_chunk_boosted.wav")
    normalize_audio(chunk_wav_path, boosted_path)

    # Voice-activity gate: only voiced chunks are note-analysed;
    # unvoiced chunks skip detection and cleanly accumulate silence.
    new_notes = (detect_melody(boosted_path)
                 if is_chunk_voiced(boosted_path) else [])

    state = interactive_session
    interrupted = False

    if new_notes:
        if pet_speaking:
            # Interruption: discard the previous melody, the next
            # reply synthesises only the new note sequence.
            interrupted = True
            state["interrupt_count"] += 1
            state["melody_log"] = [{**note} for note in new_notes]
        else:
            state["melody_log"] = extend_melody_log(state["melody_log"],
                                                    new_notes)
        state["silence_seconds"] = 0.0
    else:
        chunk_duration = _get_raw_audio_duration(chunk_wav_path)
        if chunk_duration > 0.0:
            state["silence_seconds"] += chunk_duration

    turn_ended = (state["active"]
                  and not pet_speaking
                  and len(state["melody_log"]) > 0
                  and is_user_turn_ended(state["silence_seconds"]))

    reply_filename = None
    emotion = None
    if turn_ended:
        turn_melody = state["melody_log"]
        if should_interrupt_angry(state["interrupt_count"]):
            reply_filename = "system_interactive_angry.wav"
            emotion = generate_interrupt_angry_noise(
                character=character,
                output_filename=os.path.join(RESPONSE_FOLDER, reply_filename))
            state["interrupt_count"] = 0
        else:
            reply_filename = "system_interactive_reply.wav"
            emotion = generate_interrupt_reply(
                turn_melody,
                output_filename=os.path.join(RESPONSE_FOLDER, reply_filename),
                character=character)
            add_to_memory(turn_melody, memory=short_term_memory)
        state["melody_log"] = []
        state["silence_seconds"] = 0.0

    return {
        "notes_detected": len(new_notes),
        "total_notes": len(state["melody_log"]),
        "interrupted": interrupted,
        "interrupt_count": state["interrupt_count"],
        "turn_ended": turn_ended,
        "audio_url": (f"/stream-audio/{reply_filename}"
                      if reply_filename else None),
        "emotion": emotion,
    }

# =====================================================================
# 1.6 LESSON MODE SESSION STATE
# =====================================================================
# Lesson-mode session: the user teaches a melody one section at a time.
# While a recording is running the frontend streams audio chunks; the
# detected notes accumulate into "current_notes" and — unlike
# interactive mode — NO silence threshold ends the turn: recording only
# ends when the user presses the Stop Recording button, which finalises
# the section via /api/lesson-stop. "mode" records whether the running
# recording teaches a new section ("teach") or replaces the latest one
# ("redo"); "finished" marks that the full melody has been sung once
# and the user is deciding between learning or forgetting it.
lesson_session = {
    "active": False,
    "mode": None,
    "recording": False,
    "current_notes": [],
    "sections": [],
    "finished": False,
}


def reset_lesson_session(active=False):
    """
    Resets (and optionally activates) the lesson-mode session.

    Clears the current recording buffer, all taught sections, the
    teach/redo mode, and the finished flag, setting the session active
    flag to the given value. The long-term lesson memory
    (``long_term_memory``) is deliberately NOT cleared: learned
    melodies survive across lessons.

    Args:
        active (bool, optional): Whether the session should be marked
            active after the reset. Defaults to False.

    Returns:
        None.
    """
    lesson_session.update(
        active=active,
        mode=None,
        recording=False,
        current_notes=[],
        sections=[],
        finished=False,
    )


def process_lesson_chunk(chunk_wav_path, character="default_cat"):
    """
    Processes one streamed audio chunk for the lesson-mode session.

    Mirrors the interactive chunk pipeline (normalisation, WebRTC
    voice-activity gate, melody detection) but WITHOUT the
    silence-based turn end: while the user records a section the
    detected notes are only accumulated into the current section
    buffer; the recording ends exclusively via /api/lesson-stop, so
    silence never finishes a section or triggers a reply. Chunks that
    arrive while no recording is running (e.g. while the pet sings the
    reply) are ignored.

    Args:
        chunk_wav_path (str): Path to the saved chunk audio file.
        character (str, optional): Character name matching a key in
            ``VOICE_PROFILES``. Unused for synthesis here (replies are
            synthesised on section stop), kept for pipeline symmetry.
            Defaults to "default_cat".

    Returns:
        dict: Chunk result payload with keys:
            - "notes_detected" (int): Notes found in this chunk.
            - "section_notes" (int): Notes accumulated in the current
              section recording so far.
            - "section_count" (int): Sections taught so far.
    """
    state = lesson_session
    if not state.get("recording"):
        return {
            "notes_detected": 0,
            "section_notes": len(state.get("current_notes", [])),
            "section_count": len(state.get("sections", [])),
        }

    boosted_path = os.path.join(UPLOAD_FOLDER, "lesson_chunk_boosted.wav")
    normalize_audio(chunk_wav_path, boosted_path)

    # Voice-activity gate: only voiced chunks are note-analysed.
    new_notes = (detect_melody(boosted_path)
                 if is_chunk_voiced(boosted_path) else [])

    if new_notes:
        state["current_notes"] = extend_melody_log(state["current_notes"],
                                                   new_notes)

    return {
        "notes_detected": len(new_notes),
        "section_notes": len(state["current_notes"]),
        "section_count": len(state["sections"]),
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

    During an active lesson session with at least one taught section,
    a poke is a rehearsal request instead of a poke noise: the full
    melody taught so far (all sections combined by
    :func:`combine_melody_sections`) is rendered by
    :func:`generate_lesson_reply` into the dedicated
    data/responses/system_lesson_reply.wav and sung back once — the
    click-anger logic and both memory-recall draws are skipped
    entirely. With no sections taught yet (fresh lesson), or outside
    lesson mode, the normal poke behaviour applies:

    Synthesises the click noise into data/responses/system_noise.wav:
    occasionally the Audiopet recalls a previously heard melody and
    sings it with a happy or neutral delivery, with a forced angry
    noise once the user clicks too many times. Two independent recall
    draws are used while the pet is NOT over-clicked: first the
    long-term lesson memory (with LESSON_MEMORY_SING_PROBABILITY, via
    should_sing_from_memory + pick_lesson_memory_melody), then the
    rolling short-term memory inside generate_click_noise (with its own
    MEMORY_SING_PROBABILITY); otherwise it emits a happy noise or a
    neutral click noise (0.70 happy / 0.30 none). This
    file is deliberately separate from system_reply.wav, which stays
    reserved for direct user-to-audiopet interactions.

    Returns:
        Response: JSON payload with keys:
            - "audio_url" (str): URL of the click noise (or lesson
              rehearsal) .wav.
            - "skin" (str): The active character name.
            - "emotion" (str): "happy", "none", or (once over-clicked)
              "angry" — or the lesson rehearsal delivery.
    """
    character_name = request.form.get("current_skin", "default_cat")
    try:
        click_count = int(request.form.get("click_count", 0))
    except (TypeError, ValueError):
        click_count = 0

    if (lesson_session.get("active")
            and lesson_session.get("sections")):
        # Lesson rehearsal: sing the full melody taught so far.
        combined = combine_melody_sections(lesson_session["sections"])
        reply_filename = "system_lesson_reply.wav"
        emotion = generate_lesson_reply(
            combined,
            output_filename=os.path.join(RESPONSE_FOLDER, reply_filename),
            character=character_name)
        return jsonify({
            "audio_url": f"/stream-audio/{reply_filename}",
            "skin": character_name,
            "emotion": emotion
        })

    noise_audio_filename = "system_noise.wav"
    noise_audio_path = os.path.join(RESPONSE_FOLDER, noise_audio_filename)

    over_clicked = click_count >= CLICK_NOISE_ANGER_THRESHOLD
    if (not over_clicked
            and should_sing_from_memory(
                long_term_memory,
                sing_probability=LESSON_MEMORY_SING_PROBABILITY)):
        # Lesson recall: sing a randomly chosen learned melody.
        melody_log = pick_lesson_memory_melody(long_term_memory)
        emotion = determine_noise_emotion(click_count)
        synthesise_output(melody_log,
                          output_filename=noise_audio_path,
                          sample_rate=44100,
                          character=character_name,
                          emotion=emotion)
    else:
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
    randomly chosen melody recalled from memory); once
    the count is reached the preset sad noise is delivered with the
    "sad" emotion instead and "is_final" is True — the frontend must
    stop scheduling idle noises after it. Memory recall uses two
    independent draws: first the long-term lesson memory (with
    LESSON_MEMORY_SING_PROBABILITY), then the rolling short-term memory
    inside generate_idle_noise (with IDLE_MEMORY_PROBABILITY). This
    file is deliberately separate from system_reply.wav (direct user
    interactions) and system_noise.wav (pokes).

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

    if (should_emit_idle_noise(idle_count)
            and should_sing_from_memory(
                long_term_memory,
                sing_probability=LESSON_MEMORY_SING_PROBABILITY)):
        # Lesson recall: sing a randomly chosen learned melody with the
        # same happy/neutral weighted draw as the preset idle memory
        # recall in pick_idle_noise.
        melody_log = pick_lesson_memory_melody(long_term_memory)
        emotion = str(np.random.choice(["happy", "none"], p=[0.70, 0.30]))
        synthesise_output(melody_log,
                          output_filename=idle_audio_path,
                          sample_rate=44100,
                          character=character_name,
                          emotion=emotion)
        is_final = False
    else:
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


@app.route("/api/interactive-start", methods=["POST"])
def interactive_start():
    """
    Activates the interactive-mode session.

    Resets the module-level session state and marks it active: the
    frontend can then stream audio chunks via /api/interactive-chunk.

    Returns:
        Response: JSON payload with key "active" (True).
    """
    reset_interactive_session(active=True)
    return jsonify({"active": True})


@app.route("/api/interactive-exit", methods=["POST"])
def interactive_exit():
    """
    Deactivates the interactive-mode session.

    Clears the accumulated melody, silence clock, and interrupt counter
    and marks the session inactive.

    Returns:
        Response: JSON payload with key "active" (False).
    """
    reset_interactive_session(active=False)
    return jsonify({"active": False})


@app.route("/api/interactive-chunk", methods=["POST"])
def interactive_chunk():
    """
    Handles one streamed audio chunk of the interactive-mode session.

    Expects a multipart/form-data POST with:
        - "file": the streamed audio chunk binary.
        - "current_skin" (optional): active character name; defaults
          to "default_cat" if blank.
        - "pet_speaking" (optional): "true"/"false" flag reporting
          whether the Audiopet's reply audio is currently playing.

    The chunk is processed by :func:`process_interactive_chunk`, which
    accumulates detected notes into the running turn melody, tracks
    silence, counts interruptions of the pet's singing, and — once the
    silence threshold is met — synthesises the reply (or the forced
    angry noise after too many interruptions) into a dedicated
    interactive file, deliberately separate from system_reply.wav.

    Returns:
        Response: JSON payload with keys "notes_detected",
            "total_notes", "interrupted", "interrupt_count",
            "turn_ended", "audio_url", "emotion", plus "skin".
            On failure (400): ``{"error": "<message>"}``.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file payload detected"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename property"}), 400

    character_name = request.form.get("current_skin", "default_cat")
    pet_speaking = str(request.form.get("pet_speaking", "")).lower() == "true"

    if not interactive_session.get("active"):
        return jsonify({"error": "Interactive mode is not active"}), 409

    saved_chunk_path = os.path.join(UPLOAD_FOLDER,
                                    "interactive_chunk.wav")
    audio_file.save(saved_chunk_path)

    chunk_payload = process_interactive_chunk(saved_chunk_path,
                                              pet_speaking,
                                              character=character_name)
    chunk_payload["skin"] = character_name
    return jsonify(chunk_payload)


@app.route("/api/lesson-start", methods=["POST"])
def lesson_start():
    """
    Activates the lesson-mode session.

    Resets the module-level lesson session state and marks it active:
    the frontend can then stream recording chunks via /api/lesson-chunk
    and finalise sections via /api/lesson-stop. The long-term lesson
    memory is left untouched.

    Returns:
        Response: JSON payload with key "active" (True).
    """
    reset_lesson_session(active=True)
    return jsonify({"active": True})


@app.route("/api/lesson-exit", methods=["POST"])
def lesson_exit():
    """
    Deactivates the lesson-mode session.

    Clears the current recording buffer, all taught sections, and the
    finished flag, and marks the session inactive. Learned melodies in
    the long-term lesson memory are kept.

    Returns:
        Response: JSON payload with key "active" (False).
    """
    reset_lesson_session(active=False)
    return jsonify({"active": False})


@app.route("/api/lesson-chunk", methods=["POST"])
def lesson_chunk():
    """
    Handles one streamed audio chunk of the lesson-mode recording.

    Expects a multipart/form-data POST with:
        - "file": the streamed audio chunk binary.
        - "current_skin" (optional): active character name; defaults
          to "default_cat" if blank.
        - "recording" (optional): "true"/"false" flag reporting
          whether the user is currently recording a section.
        - "mode" (optional): "teach" (new section) or "redo"
          (replace the latest section) for the running recording.

    The chunk is processed by :func:`process_lesson_chunk`, which
    accumulates detected notes into the current section buffer. Unlike
    interactive mode there is NO silence-based turn end: recording ends
    only when the frontend calls /api/lesson-stop.

    Returns:
        Response: JSON payload with keys "notes_detected",
            "section_notes", "section_count", plus "skin".
            On failure (400/409): ``{"error": "<message>"}``.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file payload detected"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename property"}), 400

    if not lesson_session.get("active"):
        return jsonify({"error": "Lesson mode is not active"}), 409

    character_name = request.form.get("current_skin", "default_cat")
    recording = str(request.form.get("recording", "true")).lower() == "true"
    mode = request.form.get("mode")

    if recording:
        if mode in ("teach", "redo"):
            lesson_session["mode"] = mode
        if not lesson_session.get("recording"):
            # A new recording pass just started: clear the section
            # buffer so the previous section's notes never leak in.
            lesson_session["current_notes"] = []
        lesson_session["recording"] = True

    saved_chunk_path = os.path.join(UPLOAD_FOLDER, "lesson_chunk.wav")
    audio_file.save(saved_chunk_path)

    chunk_payload = process_lesson_chunk(saved_chunk_path,
                                         character=character_name)
    chunk_payload["skin"] = character_name
    return jsonify(chunk_payload)


@app.route("/api/lesson-stop", methods=["POST"])
def lesson_stop():
    """
    Ends the running lesson recording and finalises the section.

    Called when the user presses the Stop Recording button (the ONLY
    way a lesson recording ends — there is no silence-based turn end).
    If the recording's melody log is empty the normal-mode failsafe
    applies: the Audiopet synthesises the preset confused noise, no
    new section is recorded ("teach" appends nothing and "redo" keeps
    the previous section via :func:`replace_last_section`), and the
    taught melody stays untouched. Otherwise the detected section
    notes either become a new section ("teach") or replace the latest
    taught section ("redo", via :func:`replace_last_section`). While
    the user is still learning, only the LATEST taught section is then
    synthesised (via :func:`generate_lesson_reply` into a dedicated
    lesson reply file) and sung back, so the feedback focuses on the
    just-taught material; the full melody can be rehearsed by poking
    the character (see /api/character-click). The recording buffer is
    cleared and the session returns to idle (sections are kept).

    Returns:
        Response: JSON payload with keys "audio_url", "skin",
            "emotion" (delivery of the latest section, or "confused"
            for the empty-recording failsafe), "total_notes"
            (full melody length), "section_count", "mode" of the
            finished recording, and "section_recorded" (False when the
            confused-noise failsafe fired).
            On failure (409): ``{"error": "<message>"}``.
    """
    if not lesson_session.get("active"):
        return jsonify({"error": "Lesson mode is not active"}), 409

    character_name = request.form.get("current_skin", "default_cat")
    mode = lesson_session.get("mode")
    current_notes = lesson_session.get("current_notes", [])

    if current_notes:
        if mode == "redo":
            lesson_session["sections"] = replace_last_section(
                lesson_session["sections"], current_notes)
        else:
            # Teach mode: only non-empty recordings become a new section.
            lesson_session["sections"].append(
                [{**note} for note in current_notes])

    lesson_session["recording"] = False
    lesson_session["mode"] = None
    lesson_session["current_notes"] = []

    reply_filename = "system_lesson_reply.wav"
    reply_path = os.path.join(RESPONSE_FOLDER, reply_filename)

    if current_notes:
        # Real section: sing the latest taught section back so the
        # feedback focuses on the just-taught material.
        latest_section = (lesson_session["sections"][-1]
                          if lesson_session["sections"] else [])
        emotion = generate_lesson_reply(
            latest_section,
            output_filename=reply_path,
            character=character_name)
    else:
        # Failsafe (same as normal recording mode): an empty melody
        # log renders the preset confused noise and no section is
        # recorded (teach appends nothing, redo keeps the previous).
        print("⚠️ Empty lesson recording: playing confused noise, "
              "no section recorded.")
        emotion = generate_lesson_reply(
            [],
            output_filename=reply_path,
            character=character_name)

    combined = combine_melody_sections(lesson_session["sections"])

    return jsonify({
        "audio_url": f"/stream-audio/{reply_filename}",
        "skin": character_name,
        "emotion": emotion,
        "total_notes": len(combined),
        "section_count": len(lesson_session["sections"]),
        "mode": mode,
        "section_recorded": bool(current_notes),
    })


@app.route("/api/lesson-finish", methods=["POST"])
def lesson_finish():
    """
    Sings the finished melody once (Finished Melody button).

    The full melody taught so far (all sections combined) is rendered
    into the dedicated lesson reply file and sung back once, and the
    session is marked "finished": the frontend then offers the Learn
    Melody / Forget Melody decision (/api/lesson-learn,
    /api/lesson-forget). Sections are kept until learn/forget resets
    the lesson.

    Returns:
        Response: JSON payload with keys "audio_url", "skin",
            "emotion", "total_notes", and "finished" (True).
            On failure (409): ``{"error": "<message>"}``.
    """
    if not lesson_session.get("active"):
        return jsonify({"error": "Lesson mode is not active"}), 409

    character_name = request.form.get("current_skin", "default_cat")
    combined = combine_melody_sections(lesson_session["sections"])
    reply_filename = "system_lesson_reply.wav"
    emotion = generate_lesson_reply(
        combined,
        output_filename=os.path.join(RESPONSE_FOLDER, reply_filename),
        character=character_name)
    lesson_session["finished"] = True

    return jsonify({
        "audio_url": f"/stream-audio/{reply_filename}",
        "skin": character_name,
        "emotion": emotion,
        "total_notes": len(combined),
        "finished": True,
    })


@app.route("/api/lesson-learn", methods=["POST"])
def lesson_learn():
    """
    Learns the finished melody into the long-term lesson memory.

    The full melody taught so far is stored in the module-level
    ``long_term_memory`` via :func:`add_to_lesson_memory` (capacity
    LESSON_MEMORY_SIZE; oldest entries evicted; empty/invalid melodies
    are never stored), then the pet rehearses the melody before the
    lesson resets to a fresh state. The rehearsal is a playback
    sequence of separately rendered pieces, returned in the
    "sequence" list for the frontend to sing in order:

    - With probability LESSON_MISTAKE_PROBABILITY the pet makes a
      mistake in the middle of the melody (one middle note mistuned by
      :func:`make_mistake_melody`), then sings the preset confused
      noise, then rehearses the entire sequence correctly a second
      time.
    - At the end of the rehearsal the pet sings the preset happy noise
      (system_lesson_happy.wav) before the frontend returns to the
      start of Lesson Mode.

    Learned melodies may later be recalled by click and idle noises
    with LESSON_MEMORY_SING_PROBABILITY.

    Returns:
        Response: JSON payload with keys "audio_url", "skin",
            "emotion" (first piece's delivery), "sequence" (ordered
            playback list of {"audio_url", "emotion"} pieces),
            "learned" (True only when a non-empty melody was actually
            stored), "memory_size" (current long-term memory size),
            and "finished" (False — fresh lesson).
            On failure (409): ``{"error": "<message>"}``.
    """
    if not lesson_session.get("active"):
        return jsonify({"error": "Lesson mode is not active"}), 409

    character_name = request.form.get("current_skin", "default_cat")
    combined = combine_melody_sections(lesson_session["sections"])
    learned = len(combined) > 0
    add_to_lesson_memory(combined, memory=long_term_memory)

    sequence = []

    if learned:
        if should_lesson_mistake():
            # Mistake in the middle of the melody: sing the mistuned
            # rendition, then a confused noise, then the correct
            # rehearsal follows below.
            print("🤦 Lesson mistake! The Audiopet fumbles the melody.")
            mistake_filename = "system_lesson_mistake.wav"
            mistake_emotion = generate_lesson_reply(
                make_mistake_melody(combined),
                output_filename=os.path.join(RESPONSE_FOLDER,
                                             mistake_filename),
                character=character_name)
            sequence.append({
                "audio_url": f"/stream-audio/{mistake_filename}",
                "emotion": mistake_emotion,
            })
            confused_filename = "system_lesson_noise.wav"
            confused_emotion = generate_lesson_noise(
                "confused",
                character=character_name,
                output_filename=os.path.join(RESPONSE_FOLDER,
                                             confused_filename))
            sequence.append({
                "audio_url": f"/stream-audio/{confused_filename}",
                "emotion": confused_emotion,
            })

        # Correct rehearsal of the entire melody.
        reply_filename = "system_lesson_reply.wav"
        emotion = generate_lesson_reply(
            combined,
            output_filename=os.path.join(RESPONSE_FOLDER, reply_filename),
            character=character_name)
        sequence.append({
            "audio_url": f"/stream-audio/{reply_filename}",
            "emotion": emotion,
        })

        # Final happy noise before the lesson returns to the start.
        happy_filename = "system_lesson_happy.wav"
        happy_emotion = generate_lesson_noise(
            "happy",
            character=character_name,
            output_filename=os.path.join(RESPONSE_FOLDER, happy_filename))
        sequence.append({
            "audio_url": f"/stream-audio/{happy_filename}",
            "emotion": happy_emotion,
        })
    else:
        # Nothing taught: confused-noise fallback only.
        reply_filename = "system_lesson_reply.wav"
        emotion = generate_lesson_reply(
            combined,
            output_filename=os.path.join(RESPONSE_FOLDER, reply_filename),
            character=character_name)
        sequence.append({
            "audio_url": f"/stream-audio/{reply_filename}",
            "emotion": emotion,
        })

    reset_lesson_session(active=True)

    return jsonify({
        "audio_url": sequence[0]["audio_url"],
        "skin": character_name,
        "emotion": sequence[0]["emotion"],
        "sequence": sequence,
        "learned": learned,
        "memory_size": len(long_term_memory),
        "finished": False,
    })


@app.route("/api/lesson-forget", methods=["POST"])
def lesson_forget():
    """
    Forgets the taught melody without learning it.

    First the pet sings the preset sad noise (``SAD_NOISE_MELODY`` with
    the "sad" emotion, via :func:`generate_lesson_noise`) into the
    dedicated lesson noise file — the frontend plays it before it
    returns the lesson UI to the fresh state. Then the lesson session
    is reset to a fresh state (sections cleared, buttons back to idle)
    without storing anything in the long-term lesson memory.

    Returns:
        Response: JSON payload with keys "audio_url" (sad noise),
            "skin", "emotion" ("sad"), "forgotten" (True) and
            "finished" (False — fresh lesson).
            On failure (409): ``{"error": "<message>"}``.
    """
    if not lesson_session.get("active"):
        return jsonify({"error": "Lesson mode is not active"}), 409

    character_name = request.form.get("current_skin", "default_cat")
    noise_filename = "system_lesson_noise.wav"
    emotion = generate_lesson_noise(
        "sad",
        character=character_name,
        output_filename=os.path.join(RESPONSE_FOLDER, noise_filename))

    reset_lesson_session(active=True)

    return jsonify({
        "audio_url": f"/stream-audio/{noise_filename}",
        "skin": character_name,
        "emotion": emotion,
        "forgotten": True,
        "finished": False,
    })


@app.route("/api/lesson-redo-noise", methods=["POST"])
def lesson_redo_noise():
    """
    Renders the noise the Audiopet makes before a Redo Previous
    Section recording starts.

    With probability LESSON_REDO_ANGRY_PROBABILITY the preset angry
    noise is picked (via :func:`pick_lesson_redo_noise`), otherwise the
    preset confused noise; either way it is rendered by
    :func:`generate_lesson_noise` into the dedicated lesson noise file
    and the frontend plays it before it starts the redo recording
    (so the pet's own noise is never recorded into the new section).
    Only available while a section has actually been taught (the redo
    button is greyed out before the first successful Teach New Notes).

    Returns:
        Response: JSON payload with keys "audio_url", "skin", and
            "emotion" ("angry" or "confused").
            On failure (409): ``{"error": "<message>"}``.
    """
    if not lesson_session.get("active"):
        return jsonify({"error": "Lesson mode is not active"}), 409
    if not lesson_session.get("sections"):
        return jsonify({"error": "No taught section to redo"}), 409

    character_name = request.form.get("current_skin", "default_cat")
    noise_kind = pick_lesson_redo_noise()
    noise_filename = "system_lesson_noise.wav"
    emotion = generate_lesson_noise(
        noise_kind,
        character=character_name,
        output_filename=os.path.join(RESPONSE_FOLDER, noise_filename))

    return jsonify({
        "audio_url": f"/stream-audio/{noise_filename}",
        "skin": character_name,
        "emotion": emotion,
    })


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
