import aubio
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

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
                duration_secs = note_frame_count * FRAME_DURATION
                melody_data.append({
                    "pitch": current_note,
                    "duration": round(duration_secs, 2)
                })
                
            # Reset counters to evaluate the brand new note pitch
            current_note = detected_pitch
            note_frame_count = 1
            
        if read < hop_size:
            # Catch the very last trailing note of the audio file before ending
            if current_note > 0 and note_frame_count >= MIN_STABLE_FRAMES:
                duration_secs = note_frame_count * FRAME_DURATION
                melody_data.append({
                    "pitch": current_note,
                    "duration": round(duration_secs, 2)
                })
            break

    for item in melody_data:
        print(f"🎵 Note: {midi_to_note_name(item['pitch']):<5} | ⏱️ Duration: {item['duration']} seconds")

    return melody_data

def midi_to_freq(midi_num):
    """Converts a MIDI note number to its frequency in Hertz."""
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))

def synthesise_output(melody_log, 
                      output_filename="synthesized_melody.wav", 
                      sample_rate=44100,
                      character="default_cat"):
    if not melody_log:
        print("\nNo stable notes found to synthesize.")
        return
        
    total_audio = []
    
    for item in melody_log:
        midi_num = item["pitch"]
        note_duration = item["duration"]
        
        freq = 440.0 * (2.0 ** ((midi_num - 69) / 12.0))
        num_samples = int(sample_rate * note_duration)
        
        t = np.linspace(0, note_duration, num_samples, endpoint=False)
        sine_wave = np.sin(2 * np.pi * freq * t)
        
        # Audio envelope to smooth out cuts and clicks
        fade_len = min(int(num_samples * 0.08), 1000)
        envelope = np.ones(num_samples)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        
        total_audio.append(sine_wave * envelope)
        
    full_audio = np.concatenate(total_audio)
    full_audio = (full_audio / np.max(np.abs(full_audio)) * 32767).astype(np.int16)
    
    wavfile.write(output_filename, sample_rate, full_audio)
    print(f"\n🎉 Successfully saved rhythm-accurate track to: {output_filename}")


