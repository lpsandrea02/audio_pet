import aubio
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import io
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

# =====================================================================
# 1. INPUT MELODY DETECTION 
# =====================================================================

def normalize_audio(input_wav, output_wav):
    """Boosts the audio file volume to its mathematical maximum limit."""
    data, sr = librosa.load(input_wav, sr=None)
    
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
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

def detect_melody(filename):
    # preferred timing setup
    samplerate = 44100
    hop_size = 1024  
    win_size = 4096 

    FRAME_DURATION = hop_size / samplerate

    source = aubio.source(filename, samplerate, hop_size)
    samplerate = source.samplerate

    pitch_detector = aubio.pitch("yinfast", win_size, hop_size, samplerate)
    pitch_detector.set_unit("midi") 
    pitch_detector.set_tolerance(0.5) 
    pitch_detector.set_silence(-45)

    # We will save dictionaries containing: {"pitch": midi_val, "duration": seconds}
    melody_data = []

    current_note = 0
    note_frame_count = 0
    MIN_STABLE_FRAMES = 5  # Filter to ignore brief accidental noise artifacts

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

    for item in melody_data:
        print(f"🎵 Note: {item['pitch']:<5} | ⏱️ Duration: {item['duration']} seconds")

    return melody_data

def midi_to_freq(midi_num):
    """Converts a MIDI note number to its frequency in Hertz."""
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))


# =====================================================================
# 2. RESPONSE SYNTHESIS
# =====================================================================

def _get_frequency(note_str):
    """Calculates exact frequency for standard notation strings like 'C5' or 'F#4'."""
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
    """Synthesizes a voice wave block that matches the exact original duration."""
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
    
    Parameters:
    - melody_log: List of {"pitch": str/int, "duration": float} dicts.
    - emotion: 'none', 'happy', 'sad', 'confused', or 'angry'.
    """
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

def _get_raw_audio_duration(input_path):
    """
    Extracts the duration (in seconds) from a raw audio input.
    Accepts either a string file path or a bytes/file-like object.
    """
    try:
        # If it's a file path string
        if isinstance(input_path, str):
            with wave.open(input_path, 'rb') as wav_file:
                return wav_file.getnframes() / float(wav_file.getframerate())
        # If it's a bytes object or an in-memory BytesIO stream
        else:
            file_stream = io.BytesIO(input_path) if isinstance(input_path, bytes) else input_path
            with wave.open(file_stream, 'rb') as wav_file:
                return wav_file.getnframes() / float(wav_file.getframerate())
    except Exception as e:
        print(f"Error reading raw audio properties: {e}. Defaulting duration comparison to 0.")
        return 0.0

def determine_emotion(melody_log, input_path, choices=None, probabilities=None):
    """
    Determines the character's emotion string by analyzing the melody log structure 
    and the real timeline of the raw input audio.
    
    Parameters:
    - melody_log: List of dicts, e.g., [{"pitch": "C5", "duration": 0.25}, ...]
    - input_path: A file path string (e.g., "recording.wav") OR raw wav bytes.
    - choices: List of available emotion strings.
    - probabilities: List of float weights matching the choices.
    """
    # Predict melody duration by summing up note lengths inside the log
    predicted_duration = sum(step.get("duration", 0.0) for step in melody_log)
    
    # Extract actual time from the raw audio input
    actual_duration = _get_raw_audio_duration(input_path)
    
    # CRITERION 1: If the response is significantly shorter than the input clip, default to confused
    if predicted_duration < (actual_duration * 0.75):
        print(f"🧐 Timeline mismatch! (Melody: {predicted_duration:.2f}s vs Raw: {actual_duration:.2f}s) -> Overriding to Confused.")
        return "confused"
    
    # Default probability distribution 
    if choices is None or probabilities is None:
        choices =       ["none", "happy", "sad", "angry"]
        probabilities = [0.70,   0.30,    0.00,  0.00] 
        
    # CRITERION 2: Manually adjustable random probability assignment
    return np.random.choice(choices, p=probabilities)
