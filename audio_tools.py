"""Audio processing tools for the Audiopet companion.

This module implements the full audio pipeline of the Audiopet:

1. Input melody detection: volume normalisation, pitch tracking with
   aubio, and melody extraction from raw user recordings.
2. Response synthesis: rendering a detected melody back as a .wav file
   in one of the predefined character voices (``VOICE_PROFILES``) and
   emotional deliveries (``EMOTION_PROFILES``).
3. Behaviour decision logic: choosing the emotion used for the response
   based on the detected melody and the raw input timeline.

Constants:
    VOICE_PROFILES (dict): Mapping from character name to voice synthesis
        settings (wave_type, pitch_scale, sub_octave, jitter,
        ring_mod_freq, animal_mod, gain).
    EMOTION_PROFILES (dict): Mapping from emotion name to delivery
        settings (vib_speed, vib_depth, pitch_envelope).
    HAPPY_NOISE_MELODY / ANGRY_NOISE_MELODY (list): Preset note
        sequences synthesised when the user pokes the character on the
        web app (rendered into system_noise.wav, never system_reply.wav).
    CLICK_NOISE_ANGER_THRESHOLD (int): Click count at which the
        click-noise draw is replaced by a forced angry noise.
    MEMORY_SIZE (int): Capacity of the rolling short-term memory: the
        number of latest successfully detected melody logs kept for
        later recall.
    MEMORY_SING_PROBABILITY (float): Probability that a click on the
        character makes the Audiopet sing a randomly chosen melody from
        short-term memory instead of its preset click noise.
"""

import aubio
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import io
import os
import shutil
import tempfile
import warnings
import wave

VOICE_PROFILES = {
    'default_cat':       {'wave_type': 'triangle', 'pitch_scale': 1.0, 'sub_octave': 0.0, 'jitter': 0.001, 'ring_mod_freq': 0,  'animal_mod': 'cat',  'gain': 0.7},
    'radio_robot':     {'wave_type': 'square',   'pitch_scale': 1.0, 'sub_octave': 0.2, 'jitter': 0.0,   'ring_mod_freq': 40, 'animal_mod': None,  'gain': 0.4},
    'tiny_dog':      {'wave_type': 'pulse',    'pitch_scale': 1.0, 'sub_octave': 0.0, 'jitter': 0.003, 'ring_mod_freq': 0,  'animal_mod': 'dog',  'gain': 0.5},
    'chubby_hamster':    {'wave_type': 'triangle', 'pitch_scale': 0.5, 'sub_octave': 0.3, 'jitter': 0.005, 'ring_mod_freq': 0,  'animal_mod': None,  'gain': 0.7},
    'evil_villain': {'wave_type': 'pulse',    'pitch_scale': 2.0, 'sub_octave': 0.0, 'jitter': 0.008, 'ring_mod_freq': 12, 'animal_mod': None,  'gain': 0.4},
    'chirp_bird':        {'wave_type': 'sine',     'pitch_scale': 2.8, 'sub_octave': 0.0, 'jitter': 0.0,   'ring_mod_freq': 0,  'animal_mod': 'bird', 'gain': 0.5}
}

EMOTION_PROFILES = {
    'none':      {'vib_speed': 0,   'vib_depth': 0.0,   'pitch_envelope': None},
    'happy':     {'vib_speed': 11,  'vib_depth': 0.012, 'pitch_envelope': 'chirp'},
    'sad':       {'vib_speed': 4.5, 'vib_depth': 0.020, 'pitch_envelope': 'whine'},
    'confused':  {'vib_speed': 7,   'vib_depth': 0.008, 'pitch_envelope': 'question'},
    'angry':     {'vib_speed': 0,   'vib_depth': 0.0,   'pitch_envelope': 'bark'}
}

# Preset melody synthesised as a "confused noise" when the detected
# note sequence is empty or invalid (e.g. the recording was too short).
CONFUSED_MELODY = [
    {'pitch': 'D5',  'duration': 0.3},
    {'pitch': 'E5',  'duration': 0.3},
    {'pitch': 'F#5', 'duration': 0.3},
    {'pitch': 'G#5', 'duration': 0.3},
]

# Preset melodies synthesised as "click noises" when the user pokes the
# Audiopet character on the web app. Most clicks produce a happy noise;
# once the click count reaches CLICK_NOISE_ANGER_THRESHOLD the pet gets
# annoyed and an angry noise is forced instead. Click noises are always
# rendered into a separate system_noise.wav so the system_reply.wav file
# stays reserved for direct user-to-audiopet interactions.
HAPPY_NOISE_MELODY = [
    {'pitch': 'C#5', 'duration': 0.2},
    {'pitch': 'F5',  'duration': 0.2},
    {'pitch': 'F#5', 'duration': 0.2},
    {'pitch': 'G#5', 'duration': 0.2},
    {'pitch': 'C#6', 'duration': 0.35},
]

ANGRY_NOISE_MELODY = [
    {'pitch': 'F5',  'duration': 0.18},
    {'pitch': 'F#5', 'duration': 0.18},
    {'pitch': 'F5',  'duration': 0.18},
    {'pitch': 'F#5', 'duration': 0.18},
]

# Number of character clicks after which the random draw is abandoned
# and an angry click noise is forced instead.
CLICK_NOISE_ANGER_THRESHOLD = 5

# =====================================================================
# SHORT TERM MEMORY
# =====================================================================

# Capacity of the rolling short-term memory (the latest n successfully
# detected melody logs are kept; customisable) and the probability that
# a click on the character triggers a memory recall instead of the
# preset click noise (controllable).
MEMORY_SIZE = 5
MEMORY_SING_PROBABILITY = 0.15

# Rolling short-term memory of the latest successfully detected melody
# logs. Confused-noise fallback cases (empty melody logs) are never
# stored; see :func:`add_to_memory`.
short_term_memory = []


def _is_valid_melody_log(melody_log):
    """
    Checks whether a melody log is storable in short-term memory.

    A valid log is a non-empty list (or tuple) of note dicts that each
    carry at least a "pitch" and a "duration" key — the exact shape
    produced by :func:`detect_melody`. Anything else (including the
    empty logs behind the confused-noise fallback, None, or malformed
    entries) is rejected.

    Args:
        melody_log (list[dict] or tuple or any): Candidate melody log.

    Returns:
        bool: True if the log can be stored in short-term memory.
    """
    if not isinstance(melody_log, (list, tuple)) or not melody_log:
        return False
    for step in melody_log:
        if (not isinstance(step, dict)
                or "pitch" not in step
                or "duration" not in step):
            return False
    return True


def add_to_memory(melody_log, capacity=MEMORY_SIZE, memory=None):
    """
    Stores a successfully detected melody log in the short-term memory.

    Only valid melody logs are stored (see
    :func:`_is_valid_melody_log`); in particular, empty logs — the
    confused-noise fallback cases — are silently ignored. The log is
    stored as a nested copy so later mutation of the caller's list or
    note dicts cannot corrupt the memory. When the memory exceeds
    ``capacity``, the oldest entries are evicted so only the latest
    ``capacity`` logs remain.

    Args:
        melody_log (list[dict]): Detected melody, one dict per note:
            {"pitch": str, "duration": float}. Invalid or empty logs
            are ignored.
        capacity (int, optional): Maximum number of logs to keep.
            Defaults to ``MEMORY_SIZE`` (5). A capacity of 0 keeps
            nothing; negative capacities keep everything.
        memory (list, optional): The rolling memory list to append to.
            Defaults to the module-level ``short_term_memory``.

    Returns:
        list: The updated memory list (the same object passed in).
    """
    if memory is None:
        memory = short_term_memory

    if not _is_valid_melody_log(melody_log):
        return memory

    # Deep-enough copy: fresh per-note dicts so the caller cannot
    # mutate memory contents after the fact (e.g. reuse of the log in
    # later pipeline stages).
    memory.append([{**step} for step in melody_log])

    if capacity >= 0 and len(memory) > capacity:
        del memory[:len(memory) - capacity]

    return memory


def pick_memory_melody(memory):
    """
    Picks a random melody log from the short-term memory for recall.

    Uses the same seeded-deterministic ``np.random`` draw structure as
    the rest of the behaviour logic.

    Args:
        memory (list): List of stored melody logs (as produced by
            :func:`add_to_memory`).

    Returns:
        list[dict]: One randomly chosen melody log, or None if the
        memory is empty or None (in which case the caller must fall
        back to the default click-noise behaviour).
    """
    if not memory:
        return None

    index = int(np.random.randint(len(memory)))
    return memory[index]


def should_sing_from_memory(memory, sing_probability=None):
    """
    Decides whether a click should trigger a memory recall instead of
    the preset click noise.

    An empty (or None) memory never sings, even at probability 1.0 —
    there is nothing to recall. Otherwise a plain ``np.random.random()``
    probability draw decides, keeping the behaviour deterministic under
    a pinned seed.

    Args:
        memory (list): List of stored melody logs.
        sing_probability (float, optional): Probability of singing a
            memory melody on a click. Defaults to
            ``MEMORY_SING_PROBABILITY``.

    Returns:
        bool: True if the click should sing a melody from memory.
    """
    if sing_probability is None:
        sing_probability = MEMORY_SING_PROBABILITY

    if not memory:
        return False

    return bool(np.random.random() < sing_probability)

# =====================================================================
# 1. INPUT MELODY DETECTION 
# =====================================================================

def normalize_audio(input_wav, output_wav):
    """
    Boosts the audio file volume to its mathematical maximum limit.

    Loads the input recording, peak-normalises it so the loudest
    sample reaches 1.0 (0 dBFS), and writes it back as 16-bit PCM.

    Args:
        input_wav (str): Path to the source audio file.
        output_wav (str): Path where the normalised .wav is written.

    Returns:
        None. Writes the normalised audio to ``output_wav``. Prints a
        warning and writes nothing if the input is completely silent,
        missing, or undecodable (fail-safe: any decode error is caught
        rather than raised, so the downstream melody pipeline can
        respond with the confused-noise fallback instead of crashing).
    """
    # Remove any stale output first so a failed/skipped normalisation
    # can never leave an old recording behind for detect_melody() to
    # re-analyse.
    if os.path.exists(output_wav):
        os.remove(output_wav)

    try:
        with warnings.catch_warnings():
            # Compressed containers (browser webm/ogg uploads) decode
            # through audioread's deprecated fallback; silence the
            # deprecation noise so the pipeline log stays readable.
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            data, sr = librosa.load(input_wav, sr=None)
    except Exception as e:
        reason = str(e).strip() or type(e).__name__
        print(f"⚠️ Could not decode '{input_wav}': {reason}. Skipping normalisation.")
        return
    
    # Convert integer PCM data to float for calculations
    float_data = data.astype(np.float32)
    
    # Find the loudest peak in your audio file
    max_peak = np.max(np.abs(float_data))
    
    if max_peak > 0:
        # Scale the entire audio timeline so the loudest point hits 1.0 (0dB max)
        normalized_data = float_data / max_peak
        # Convert back to standard 16-bit audio
        final_data = (normalized_data * 32767).astype(np.int16)
        wavfile.write(output_wav, sr, final_data)
        print("🔊 Audio successfully boosted and normalized!")
    else:
        print("⚠️ File is completely silent.")


def midi_to_note_name(midi_num):
    """
    Converts a MIDI note number to standard scientific pitch notation.

    Args:
        midi_num (int): MIDI note number (e.g. 69 for A4).

    Returns:
        str: Note name in the form ``"<letter><accidental><octave>"``
        (e.g. "A4", "C#3").
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

def detect_melody(filename):
    """
    Detects the melody sequence of a recording using aubio pitch tracking.

    Streams the audio through the "yinfast" pitch detector frame by
    frame, groups consecutive frames of the same pitch into notes, and
    filters out blips shorter than ``MIN_STABLE_FRAMES`` frames.

    Args:
        filename (str): Path to the audio file to analyse.

    Returns:
        list[dict]: Detected melody, one dict per note:
            - "pitch" (str): Note name in scientific pitch notation
              (e.g. "D5"), as produced by :func:`midi_to_note_name`.
            - "duration" (float): Note length in seconds, rounded to
              2 decimal places (includes a fixed +0.1s tail).
        Returns an empty list if no stable note is detected, or if the
        file is missing, invalid, or corrupted (fail-safe: any aubio
        error is caught rather than raised).
    """
    # preferred timing setup
    samplerate = 44100
    hop_size = 1024  
    win_size = 4096 

    FRAME_DURATION = hop_size / samplerate

    # We will save dictionaries containing: {"pitch": midi_val, "duration": seconds}
    melody_data = []

    current_note = 0
    note_frame_count = 0
    MIN_STABLE_FRAMES = 5  # Filter to ignore brief accidental noise artifacts

    try:
        source = aubio.source(filename, samplerate, hop_size)
        samplerate = source.samplerate

        pitch_detector = aubio.pitch("yinfast", win_size, hop_size, samplerate)
        pitch_detector.set_unit("midi") 
        pitch_detector.set_tolerance(0.5) 
        pitch_detector.set_silence(-45)

        while True:
            samples, read = source()
            pitch = pitch_detector(samples)
            confidence = pitch_detector.get_confidence()
            
            # Bypasses NumPy 1.25 warning safely
            raw_pitch_val = pitch
            detected_pitch = int(np.round(raw_pitch_val)) if confidence > 0.5 else 0

            if detected_pitch == current_note:
                # Note is being held, increment frame duration counter
                note_frame_count += 1
            else:
                # The note changed or silence occurred. 
                # Save the previous note if it lasted long enough to be real.
                if current_note > 0 and note_frame_count >= MIN_STABLE_FRAMES:
                    duration_secs = note_frame_count * FRAME_DURATION + 0.1
                    melody_data.append({
                        "pitch": midi_to_note_name(current_note),
                        "duration": round(duration_secs, 2)
                    })
                    
                # Reset counters to evaluate the brand new note pitch
                current_note = detected_pitch
                note_frame_count = 1
                
            if read < hop_size:
                # Catch the very last trailing note of the audio file before ending
                if current_note > 0 and note_frame_count >= MIN_STABLE_FRAMES:
                    duration_secs = note_frame_count * FRAME_DURATION + 0.1
                    melody_data.append({
                        "pitch": midi_to_note_name(current_note),
                        "duration": round(duration_secs, 2)
                    })
                break
    except Exception as e:
        # Invalid, truncated, corrupted, or missing audio must never
        # crash the pipeline; report and continue with whatever (if
        # anything) was detected before the failure.
        print(f"⚠️ Melody detection failed for '{filename}': {e}. Returning no notes.")

    for item in melody_data:
        print(f"🎵 Note: {item['pitch']:<5} | ⏱️ Duration: {item['duration']} seconds")

    return melody_data

def midi_to_freq(midi_num):
    """
    Converts a MIDI note number to its frequency in Hertz.

    Args:
        midi_num (int or float): MIDI note number (69 = A4 = 440 Hz).

    Returns:
        float: Frequency of the note in Hz using equal temperament
        tuned to A4 = 440 Hz.
    """
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))


# =====================================================================
# 2. RESPONSE SYNTHESIS
# =====================================================================

def _get_frequency(note_str):
    """
    Calculates exact frequency for standard notation strings like 'C5' or 'F#4'.

    Args:
        note_str (str, int, or float): Note in scientific pitch notation
            (e.g. "C5", "F#4"), or a numeric frequency in Hz, or one of
            the rest markers "rest", "", "0".

    Returns:
        float: Frequency of the note in Hz (equal temperament,
        A4 = 440 Hz), the number itself if numeric input was given,
        or 0.0 for rests and unparseable input.
    """
    if isinstance(note_str, (int, float)):
        return float(note_str)
    note_str = str(note_str).strip()
    if note_str in ['rest', '', '0']:
        return 0.0
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        note_name = note_str[:-1]
        octave = int(note_str[-1])
        semitones_from_c4 = chromatic_scale.index(note_name) + (octave - 4) * 12
        c4_freq = 440 * (2 ** (-9 / 12)) 
        return c4_freq * (2 ** (semitones_from_c4 / 12))
    except (ValueError, IndexError):
        return 0.0


def _generate_tone_with_emotion(base_freq, duration, voice_cfg, emotion_cfg, is_last_note, sample_rate):
    """
    Synthesizes a voice wave block that matches the exact original duration.

    Renders one note as a waveform shaped by the character voice
    (timbre, sub-octave, jitter, ring modulation, animal pitch
    overrides) and the emotion delivery (vibrato and pitch envelope),
    then applies a short legato attack/release window.

    Args:
        base_freq (float): Fundamental frequency of the note in Hz
            (0.0 produces silence for the given duration).
        duration (float): Note duration in seconds; the output block
            always contains exactly ``sample_rate * duration`` samples.
        voice_cfg (dict): Voice settings from ``VOICE_PROFILES`` for
            the active character.
        emotion_cfg (dict): Emotion settings from ``EMOTION_PROFILES``
            for the active emotion.
        is_last_note (bool): True if this is the final note of the
            melody (enables the 'question' pitch envelope tail).
        sample_rate (int): Output sample rate in Hz.

    Returns:
        numpy.ndarray: Float waveform samples for the note, scaled by
        the voice's gain, with the same length in seconds as
        ``duration``.
    """
    if base_freq == 0:
        return np.zeros(int(sample_rate * duration))
        
    freq = base_freq * voice_cfg['pitch_scale']
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. Base Vibrato (from Emotion Profile)
    inst_freq = np.ones_like(t) * freq
    if emotion_cfg['vib_depth'] > 0:
        vibrato = 1.0 + emotion_cfg['vib_depth'] * np.sin(2 * np.pi * emotion_cfg['vib_speed'] * t)
        inst_freq *= vibrato
        
    # 2. Emotional Pitch Curves
    env_type = emotion_cfg['pitch_envelope']
    if env_type == 'chirp':
        inst_freq *= (1.0 + 0.05 * np.exp(-t * 40))
    elif env_type == 'whine':
        inst_freq *= (1.0 - 0.06 * (t / duration))
    elif env_type == 'question' and is_last_note:
        inst_freq *= (1.0 + 0.30 * (t / duration)**3)
    elif env_type == 'bark':
        inst_freq *= (1.0 + 0.10 * np.exp(-t * 60))
        
    # 3. Native Animal Character Overrides
    if voice_cfg['animal_mod'] == 'cat':
        inst_freq *= (1.0 + 0.08 * np.sin(np.pi * (t / duration)) - 0.04 * (t / duration))
    elif voice_cfg['animal_mod'] == 'dog':
        inst_freq *= (1.0 + 0.15 * np.exp(-t * 40))
    elif voice_cfg['animal_mod'] == 'bird':
        inst_freq *= (0.98 + 0.08 * (t / duration)**2)
        
    # 4. Organic Voice Micro-Tremor
    if voice_cfg['jitter'] > 0:
        inst_freq *= (1.0 + np.random.uniform(-voice_cfg['jitter'], voice_cfg['jitter'], len(t)))
        
    phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
    
    # 5. Timbre Synthesis
    if voice_cfg['wave_type'] == 'sine':
        wave = np.sin(phase)
    elif voice_cfg['wave_type'] == 'triangle':
        wave = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
    elif voice_cfg['wave_type'] == 'square':
        wave = np.sign(np.sin(phase))
    elif voice_cfg['wave_type'] == 'pulse':
        wave = np.where((phase % (2 * np.pi)) < (2 * np.pi * 0.25), 1.0, -1.0)
        
    # Sub-octaves
    if voice_cfg['sub_octave'] > 0:
        sub_phase = phase / 2.0
        sub_wave = np.sign(np.sin(sub_phase)) if voice_cfg['wave_type'] == 'square' else np.sin(sub_phase)
        wave = (wave * (1.0 - voice_cfg['sub_octave'])) + (sub_wave * voice_cfg['sub_octave'])
        
    # Ring mod
    if voice_cfg['ring_mod_freq'] > 0:
        wave *= np.sin(2 * np.pi * voice_cfg['ring_mod_freq'] * t)
        
    # 6. Envelope (Legato Style window to respect absolute timing)
    env = np.ones_like(t)
    attack_samples = int(0.015 * sample_rate)
    release_samples = int(0.015 * sample_rate)
    
    if len(t) > (attack_samples + release_samples):
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        env[-release_samples:] = np.linspace(1, 0, release_samples)
        
    return wave * env * voice_cfg['gain']


def synthesise_output(melody_log, 
                      output_filename="synthesized_melody.wav", 
                      sample_rate=44100, 
                      character="default_cat",
                      emotion="none"):
    """
    Synthesizes structural .wav sequences from logs incorporating separate character and emotional modifiers.

    Renders each note of the melody log as a tone with the chosen
    character voice and emotional delivery, concatenates them, and
    writes the result as a 16-bit PCM .wav file. If the melody log is
    empty or None, the preset confused noise (``CONFUSED_MELODY``) is
    synthesised with the "confused" emotion instead.

    Parameters:
    - melody_log (list[dict]): List of {"pitch": str/int, "duration": float} dicts.
    - output_filename (str, optional): Path of the .wav file to write.
      Defaults to "synthesized_melody.wav".
    - sample_rate (int, optional): Output sample rate in Hz.
      Defaults to 44100.
    - character (str, optional): Key into ``VOICE_PROFILES``; falls
      back to "default_cat" if unknown. Defaults to "default_cat".
    - emotion (str, optional): One of 'none', 'happy', 'sad',
      'confused', or 'angry'; falls back to "none" if unknown.
      Defaults to "none".

    Returns:
        str: The emotion actually used for the synthesis, so callers
        can detect when the confused-noise fallback fired ("confused"
        instead of the passed-in emotion).

    Raises:
        ValueError: If a non-empty ``melody_log`` fails to render.
    """
    # Fail-safe: empty or missing melody log -> synthesise the preset
    # confused noise instead of crashing on an empty note sequence.
    if not melody_log:
        print("melody_log is empty, generating confused noise…")
        melody_log = CONFUSED_MELODY
        emotion = "confused"
        
    voice_cfg = VOICE_PROFILES.get(character, VOICE_PROFILES['default_cat'])
    emotion_cfg = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES['none'])
    
    audio_buffer = []
    total_notes = len(melody_log)
    
    for idx, step in enumerate(melody_log):
        note_str = step.get("pitch", "rest")
        duration = step.get("duration", 0.2)
        
        base_freq = _get_frequency(note_str)
        is_last_note = (idx == total_notes - 1)
        
        # Pass both matrices down the pipe seamlessly
        tone = _generate_tone_with_emotion(
            base_freq, duration, voice_cfg, emotion_cfg, is_last_note, sample_rate
        )
        audio_buffer.append(tone)
        
    audio_signal = np.concatenate(audio_buffer)
    
    if np.max(np.abs(audio_signal)) > 1.0:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))
        
    audio_int16 = np.int16(audio_signal * 32767)
    wavfile.write(output_filename, sample_rate, audio_int16)
    print(f"💾 Rendered -> {output_filename} ({character} + {emotion})")
    return emotion

def _wave_duration(source):
    """
    Reads duration (in seconds) from a WAV source via the built-in
    ``wave`` module (RIFF/WAV only).

    Args:
        source (str or file-like): Path to a .wav file, or an open
        binary stream positioned at the start of the RIFF data.

    Returns:
        float: Audio duration in seconds.

    Raises:
        Any ``wave.Error``/I/O failure propagates to the caller.
    """
    with wave.open(source, 'rb') as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def _librosa_duration_from_file(path):
    """
    Reads duration (in seconds) from any audio file librosa can decode
    (wav, webm, ogg, mp3, flac, ... via soundfile/audioread).

    Args:
        path (str): Path to the audio file.

    Returns:
        float: Audio duration in seconds.

    Raises:
        Exception: Propagates if no decoder could handle the file.
    """
    with warnings.catch_warnings():
        # audioread's ffmpeg fallback is deprecated but is currently
        # the only route that decodes browser webm uploads; silence
        # the noise so the pipeline log stays readable.
        warnings.simplefilter("ignore", FutureWarning)
        return float(librosa.get_duration(path=path))


def _get_raw_audio_duration(input_path):
    """
    Extracts the duration (in seconds) from a raw audio input.
    Accepts either a string file path or a bytes/file-like object.

    Uses the fast built-in ``wave`` reader first (RIFF/WAV only). If
    the input is not a RIFF file (e.g. a browser MediaRecorder webm
    upload mislabeled as .wav), it falls back to librosa, which
    decodes compressed containers such as webm, ogg, mp3, and flac.

    Args:
        input_path (str, bytes, or file-like): Path to an audio file,
            raw audio bytes, or an in-memory stream.

    Returns:
        float: Audio duration in seconds, or 0.0 if the audio could
        not be decoded by any available reader.
    """
    try:
        if isinstance(input_path, str):
            try:
                return _wave_duration(input_path)
            except Exception:
                return _librosa_duration_from_file(input_path)

        # bytes object or in-memory stream: the wave reader can try
        # it directly, but compressed containers must be materialised
        # to a temp file for the ffmpeg-based fallback to read.
        file_stream = io.BytesIO(input_path) if isinstance(input_path, bytes) else input_path
        try:
            return _wave_duration(file_stream)
        except Exception:
            position = None
            try:
                position = file_stream.tell()
                file_stream.seek(0)
            except (AttributeError, OSError):
                pass
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                shutil.copyfileobj(file_stream, tmp)
                tmp_path = tmp.name
            try:
                return _librosa_duration_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)
    except Exception as e:
        print(f"Could not determine raw audio duration ({e}). "
              "Defaulting duration comparison to 0.")
        return 0.0

def determine_emotion(melody_log, input_path, choices=None, probabilities=None):
    """
    Determines the character's emotion string by analyzing the melody log structure 
    and the real timeline of the raw input audio.

    First checks for a timeline mismatch: if the total detected melody
    duration is under 75% of the raw recording length, the emotion is
    forced to "confused". Otherwise, an emotion is drawn at random
    from the given choices with the given probability weights.

    Fail-safe behaviour:
    - An empty or None ``melody_log`` always yields "confused"
      (mirroring the confused-noise fallback in :func:`synthesise_output`).
    - If the raw input's duration cannot be read at all (0.0), the
      timeline check cannot run, so the weighted random draw decides
      and the caveat is logged.

    Parameters:
    - melody_log (list[dict]): List of dicts, e.g.,
      [{"pitch": "C5", "duration": 0.25}, ...]
    - input_path (str or bytes): A file path string
      (e.g. "recording.wav") OR raw audio bytes.
    - choices (list[str], optional): List of available emotion strings.
      Defaults to ["none", "happy", "sad", "angry"].
    - probabilities (list[float], optional): List of float weights
      matching the choices. Defaults to [0.70, 0.30, 0.00, 0.00].

    Returns:
        str: The chosen emotion, either "confused" (timeline mismatch)
        or one of ``choices`` sampled according to ``probabilities``.
    """
    # Fail-safe: no detected melody at all is always a confused
    # response, even if the raw input itself cannot be decoded.
    if not melody_log:
        return "confused"

    # Predict melody duration by summing up note lengths inside the log,
    # tolerating missing or invalid per-note durations
    predicted_duration = 0.0
    for step in melody_log:
        try:
            predicted_duration += float(step.get("duration") or 0.0)
        except (TypeError, ValueError, AttributeError):
            continue

    # Extract actual time from the raw audio input
    actual_duration = _get_raw_audio_duration(input_path)

    # CRITERION 1: If the response is significantly shorter than the input clip, default to confused
    if actual_duration > 0.0 and predicted_duration < (actual_duration * 0.4):
        print(f"🧐 Timeline mismatch! (Melody: {predicted_duration:.2f}s vs Raw: {actual_duration:.2f}s) -> Overriding to Confused.")
        return "confused"

    if actual_duration <= 0.0:
        print("⚠️ Raw input duration unreadable; timeline check skipped, using default emotion draw.")

    # Default probability distribution 
    if choices is None or probabilities is None:
        choices =       ["none", "happy", "sad", "angry"]
        probabilities = [0.70,   0.30,    0.00,  0.00] 
        
    # CRITERION 2: Manually adjustable random probability assignment.
    # np.random.choice returns numpy str_; coerce to a plain str so
    # downstream dict lookups (EMOTION_PROFILES, Flask JSON payload)
    # behave like ordinary strings.
    return str(np.random.choice(choices, p=probabilities))


def determine_noise_emotion(click_count):
    """
    Determines which click noise the Audiopet makes when the user pokes
    the character on the web app.

    Uses the same weighted ``np.random.choice`` probability structure as
    the standard responses in :func:`determine_emotion`, restricted to
    the noise choices: while the pet has not been over-clicked the draw
    is 0.70 happy / 0.30 none (a neutral click delivery). Once
    ``click_count`` reaches ``CLICK_NOISE_ANGER_THRESHOLD`` the random
    draw is abandoned and an angry noise is forced.

    Parameters:
    - click_count (int): How many times the user has poked the
      character (the frontend counts consecutive clicks).

    Returns:
        str: "happy", "none", or (once over-clicked) "angry".
    """
    if click_count >= CLICK_NOISE_ANGER_THRESHOLD:
        print(f"😠 Clicked {click_count} times! The Audiopet is fed up.")
        return "angry"

    choices =       ["happy", "none"]
    probabilities = [0.70,     0.30]

    # Same random probability structure as determine_emotion(): a plain
    # np.random.choice, coerced to str for downstream dict lookups.
    return str(np.random.choice(choices, p=probabilities))


def generate_click_noise(character="default_cat", click_count=0,
                         output_filename="system_noise.wav",
                         memory=None, sing_probability=None):
    """
    Synthesises the click noise for a poke on the character and writes
    it to a dedicated noise file.

    Occasionally (with a controllable probability) the Audiopet recalls
    a randomly chosen melody from its short-term memory and sings that
    back instead, with a happy or neutral delivery. Memory recall only
    happens while the pet is NOT over-clicked: once ``click_count``
    reaches ``CLICK_NOISE_ANGER_THRESHOLD`` only the preset angry noise
    plays (never a memory melody) until the click count resets (e.g.
    after a new user recording). If the memory is empty (or the
    probability draw fails) the behaviour falls back to the default
    click noise: the emotion is decided via
    :func:`determine_noise_emotion` (0.70 happy / 0.30 none, forced
    angry once over-clicked) and the preset melody is rendered — the
    angry noise (``ANGRY_NOISE_MELODY``) once over-clicked, otherwise
    the happy noise (``HAPPY_NOISE_MELODY``) in the drawn delivery.
    Unlike the standard response pipeline,
    the result must never be written to ``system_reply.wav``: that file
    is reserved for direct user-to-audiopet interactions, so callers
    pass a separate path (e.g. ``data/responses/system_noise.wav``).

    Parameters:
    - character (str, optional): Key into ``VOICE_PROFILES``; falls
      back to "default_cat" if unknown. Defaults to "default_cat".
    - click_count (int, optional): How many times the user has poked
      the character. Defaults to 0.
    - output_filename (str, optional): Path of the .wav file to write.
      Defaults to "system_noise.wav".
    - memory (list, optional): The short-term memory (list of stored
      melody logs, as produced by :func:`add_to_memory`). An empty or
      None memory disables memory recall entirely. Defaults to None.
    - sing_probability (float, optional): Probability of singing a
      memory melody on this click. Defaults to
      ``MEMORY_SING_PROBABILITY``.

    Returns:
        str: The noise emotion actually synthesised ("happy", "none",
        or "angry"), for the frontend animation.
    """
    over_clicked = click_count >= CLICK_NOISE_ANGER_THRESHOLD
    if (not over_clicked
            and should_sing_from_memory(memory,
                                        sing_probability=sing_probability)):
        # Memory recall: sing a randomly chosen previously heard melody
        melody_log = pick_memory_melody(memory)
        emotion = determine_noise_emotion(click_count)
    else:
        emotion = determine_noise_emotion(click_count)
        melody_log = (ANGRY_NOISE_MELODY if over_clicked
                      else HAPPY_NOISE_MELODY)

    synthesise_output(melody_log,
                      output_filename=output_filename,
                      sample_rate=44100,
                      character=character,
                      emotion=emotion)
    return emotion
