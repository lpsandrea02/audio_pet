"""Unit tests for audio_tools.py.

Tests both existing behaviour (baseline), the confused-noise fallback
behaviour (detect_melody returning [] on invalid input, and
synthesise_output generating a confused melody when given an empty log),
and the click-noise API for the clickable character feature (preset
happy/none click noise melodies, click-count driven emotion draw, and
generation into a separate system_noise.wav that never touches
system_reply.wav).

Run with:  python3 test_audio_tools.py -v
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
import scipy.io.wavfile as wavfile

FFMPEG = shutil.which("ffmpeg")

from audio_tools import (
    normalize_audio,
    detect_melody,
    midi_to_note_name,
    midi_to_freq,
    _get_frequency,
    _generate_tone_with_emotion,
    synthesise_output,
    _get_raw_audio_duration,
    determine_emotion,
    VOICE_PROFILES,
    EMOTION_PROFILES,
)

SAMPLE_RATE = 44100


# ---------------------------------------------------------------------
# Helpers for generating test audio
# ---------------------------------------------------------------------
def make_tone(freq, duration, amp=0.5, sample_rate=SAMPLE_RATE):
    """Return a sine tone of the given duration as a float array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


def write_wav(path, samples, sample_rate=SAMPLE_RATE):
    """Write float samples to a 16-bit PCM wav file."""
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, data)


def read_wav(path):
    """Read a wav file, returning (sample_rate, float samples)."""
    sr, data = wavfile.read(path)
    return sr, data.astype(np.float32) / 32767.0


class AudioToolsTestCase(unittest.TestCase):
    """Base class providing a temp dir with pre-generated test files."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="audiopet_test_")

        # Quiet recording (peak ~0.01) for normalisation testing
        cls.quiet_wav = os.path.join(cls.tmpdir, "quiet.wav")
        write_wav(cls.quiet_wav, make_tone(440.0, 0.5, amp=0.01))

        # Completely silent recording
        cls.silent_wav = os.path.join(cls.tmpdir, "silent.wav")
        write_wav(cls.silent_wav, np.zeros(SAMPLE_RATE // 2))

        # Valid melody: C5, E5, G5 held for 0.4s each (~1.2s total)
        cls.melody_wav = os.path.join(cls.tmpdir, "melody.wav")
        melody_samples = np.concatenate([
            make_tone(523.25, 0.4),  # C5
            make_tone(659.25, 0.4),  # E5
            make_tone(783.99, 0.4),  # G5
        ])
        write_wav(cls.melody_wav, melody_samples)

        # Corrupted file: random garbage bytes with a .wav extension
        cls.corrupt_wav = os.path.join(cls.tmpdir, "corrupt.wav")
        with open(cls.corrupt_wav, "wb") as f:
            f.write(os.urandom(2048))

        # Header-only wav (no data chunk payload): simulates stopping
        # recording immediately after starting
        with wave.open(os.path.join(cls.tmpdir, "header_only.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
        cls.header_only_wav = os.path.join(cls.tmpdir, "header_only.wav")

        # Extremely short but valid recording (~20ms)
        cls.tiny_wav = os.path.join(cls.tmpdir, "tiny.wav")
        write_wav(cls.tiny_wav, make_tone(440.0, 0.02))

        # One-second recording, used for duration/emotion comparisons
        cls.one_sec_wav = os.path.join(cls.tmpdir, "one_sec.wav")
        write_wav(cls.one_sec_wav, make_tone(440.0, 1.0))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    @classmethod
    def make_webm(cls, name, duration=1.0, freq=440.0):
        """Generate a real webm audio file via ffmpeg, named with a .wav
        extension to replicate the browser's mislabeled MediaRecorder
        upload (Blob type "audio/wav" wrapping webm bytes)."""
        target = os.path.join(cls.tmpdir, name)
        encoded = os.path.join(cls.tmpdir, name + ".encoded.webm")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"sine=frequency={freq}:duration={duration}",
             "-c:a", "libopus", encoded],
            capture_output=True, check=True)
        os.replace(encoded, target)
        return target


# ---------------------------------------------------------------------
# Note utility functions
# ---------------------------------------------------------------------
class TestNoteUtilities(AudioToolsTestCase):

    def test_midi_to_note_name(self):
        self.assertEqual(midi_to_note_name(69), "A4")
        self.assertEqual(midi_to_note_name(60), "C4")
        self.assertEqual(midi_to_note_name(61), "C#4")
        self.assertEqual(midi_to_note_name(21), "A0")

    def test_midi_to_freq(self):
        self.assertAlmostEqual(midi_to_freq(69), 440.0, places=6)
        self.assertAlmostEqual(midi_to_freq(60), 261.6256, places=4)

    def test_get_frequency_note_names(self):
        self.assertAlmostEqual(_get_frequency("A4"), 440.0, places=6)
        self.assertAlmostEqual(_get_frequency("C4"), 261.6256, places=4)
        self.assertAlmostEqual(_get_frequency("F#4"), 369.9944, places=4)

    def test_get_frequency_rest_markers(self):
        for marker in ("rest", "", "0"):
            self.assertEqual(_get_frequency(marker), 0.0)

    def test_get_frequency_numeric_input(self):
        self.assertEqual(_get_frequency(440), 440.0)
        self.assertEqual(_get_frequency(523.25), 523.25)

    def test_get_frequency_invalid_note(self):
        self.assertEqual(_get_frequency("X9"), 0.0)

    def test_get_frequency_click_noise_preset_notes(self):
        # All note names used by the upcoming click-noise preset
        # melodies (happy + angry) must parse to audible frequencies.
        # BASELINE: these exact notes must already be parseable before
        # any modification to audio_tools.py.
        expected = {
            "C#5": 554.3653,
            "F5": 698.4565,
            "F#5": 739.9888,
            "G#5": 830.6094,
            "C#6": 1108.7305,
        }
        for note, freq in expected.items():
            parsed = _get_frequency(note)
            self.assertAlmostEqual(parsed, freq, places=3,
                                   msg=f"preset note {note} did not parse")
            self.assertGreater(parsed, 0.0)


# ---------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------
class TestNormalizeAudio(AudioToolsTestCase):

    def test_quiet_audio_boosted_to_full_scale(self):
        out_path = os.path.join(self.tmpdir, "quiet_boosted.wav")
        normalize_audio(self.quiet_wav, out_path)
        self.assertTrue(os.path.exists(out_path))
        sr, data = read_wav(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertAlmostEqual(np.max(np.abs(data)), 1.0, places=2)

    def test_output_is_16bit_pcm(self):
        out_path = os.path.join(self.tmpdir, "pcm_check.wav")
        normalize_audio(self.quiet_wav, out_path)
        with wave.open(out_path, "rb") as f:
            self.assertEqual(f.getsampwidth(), 2)

    def test_silent_audio_writes_nothing(self):
        out_path = os.path.join(self.tmpdir, "silent_boosted.wav")
        normalize_audio(self.silent_wav, out_path)
        self.assertFalse(os.path.exists(out_path))

    def test_undecodable_input_writes_nothing(self):
        # NEW behaviour: garbage input fail-safes instead of raising
        # (baseline: librosa.load raises NoBackendError, crashing the
        # Flask request in app.py).
        out_path = os.path.join(self.tmpdir, "corrupt_boosted.wav")
        normalize_audio(self.corrupt_wav, out_path)
        self.assertFalse(os.path.exists(out_path))

    def test_missing_input_writes_nothing(self):
        out_path = os.path.join(self.tmpdir, "missing_boosted.wav")
        normalize_audio(os.path.join(self.tmpdir, "nope.wav"), out_path)
        self.assertFalse(os.path.exists(out_path))

    def test_stale_output_removed_on_undecodable_input(self):
        # A previous valid normalisation must not linger: stale output
        # would otherwise be re-analysed by detect_melody() downstream.
        out_path = os.path.join(self.tmpdir, "stale_boosted.wav")
        normalize_audio(self.melody_wav, out_path)
        self.assertTrue(os.path.exists(out_path))
        normalize_audio(self.corrupt_wav, out_path)
        self.assertFalse(os.path.exists(out_path))


# ---------------------------------------------------------------------
# Tone generation
# ---------------------------------------------------------------------
class TestGenerateTone(AudioToolsTestCase):

    def test_output_length_matches_duration(self):
        tone = _generate_tone_with_emotion(
            440.0, 0.5, VOICE_PROFILES["default_cat"],
            EMOTION_PROFILES["none"], False, SAMPLE_RATE)
        self.assertEqual(len(tone), int(SAMPLE_RATE * 0.5))

    def test_zero_freq_produces_silence(self):
        tone = _generate_tone_with_emotion(
            0.0, 0.3, VOICE_PROFILES["default_cat"],
            EMOTION_PROFILES["happy"], False, SAMPLE_RATE)
        self.assertEqual(len(tone), int(SAMPLE_RATE * 0.3))
        self.assertTrue(np.all(tone == 0))

    def test_emotion_and_voice_configs_apply(self):
        # Every supported voice/emotion combination must render without
        # error and produce a finite, bounded signal
        for char, voice_cfg in VOICE_PROFILES.items():
            for emot, emotion_cfg in EMOTION_PROFILES.items():
                tone = _generate_tone_with_emotion(
                    440.0, 0.2, voice_cfg, emotion_cfg, True, SAMPLE_RATE)
                self.assertEqual(len(tone), int(SAMPLE_RATE * 0.2))
                self.assertTrue(np.all(np.isfinite(tone)))
                self.assertLessEqual(np.max(np.abs(tone)), 1.0)


# ---------------------------------------------------------------------
# Melody detection
# ---------------------------------------------------------------------
class TestDetectMelody(AudioToolsTestCase):

    def test_valid_melody_detected(self):
        melody = detect_melody(self.melody_wav)
        self.assertGreater(len(melody), 0)
        for note in melody:
            self.assertIn("pitch", note)
            self.assertIn("duration", note)
            self.assertIsInstance(note["pitch"], str)
            self.assertIsInstance(note["duration"], float)
            self.assertGreater(note["duration"], 0)

    def test_tiny_recording_returns_empty_list(self):
        # Recording stopped almost immediately after starting
        self.assertEqual(detect_melody(self.tiny_wav), [])

    def test_corrupted_file_returns_empty_list(self):
        # NEW behaviour: fail-safe instead of raising.
        # BASELINE: aubio raises an exception for garbage input.
        self.assertEqual(detect_melody(self.corrupt_wav), [])

    def test_header_only_file_returns_empty_list(self):
        # NEW behaviour: fail-safe instead of raising.
        self.assertEqual(detect_melody(self.header_only_wav), [])

    def test_missing_file_returns_empty_list(self):
        # NEW behaviour: fail-safe instead of raising.
        missing = os.path.join(self.tmpdir, "does_not_exist.wav")
        self.assertEqual(detect_melody(missing), [])


# ---------------------------------------------------------------------
# Response synthesis
# ---------------------------------------------------------------------
class TestSynthesiseOutput(AudioToolsTestCase):

    def test_valid_melody_renders_wav(self):
        melody_log = [
            {"pitch": "C5", "duration": 0.3},
            {"pitch": "E5", "duration": 0.3},
        ]
        out_path = os.path.join(self.tmpdir, "reply.wav")
        result = synthesise_output(melody_log, out_path,
                                   character="default_cat", emotion="happy")
        self.assertTrue(os.path.exists(out_path))
        sr, data = wavfile.read(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        expected_frames = int(0.3 * SAMPLE_RATE) * 2
        self.assertEqual(len(data), expected_frames)
        self.assertEqual(result, "happy")

    def test_unknown_character_falls_back_to_default(self):
        out_path = os.path.join(self.tmpdir, "unknown_char.wav")
        result = synthesise_output(
            [{"pitch": "C5", "duration": 0.3}], out_path,
            character="not_a_character", emotion="none")
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(result, "none")

    def test_empty_melody_generates_confused_noise(self):
        # NEW behaviour: empty log -> preset confused melody synthesised.
        # BASELINE: raises ValueError.
        out_path = os.path.join(self.tmpdir, "confused_reply.wav")
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            result = synthesise_output([], out_path,
                                       character="default_cat",
                                       emotion="happy")
        output = stdout_capture.getvalue()
        self.assertIn("melody_log is empty", output)
        self.assertIn("confused noise", output)
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(result, "confused")

    def test_none_melody_generates_confused_noise(self):
        out_path = os.path.join(self.tmpdir, "confused_none.wav")
        result = synthesise_output(None, out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(result, "confused")

    def test_confused_noise_writes_audible_signal(self):
        out_path = os.path.join(self.tmpdir, "confused_audible.wav")
        synthesise_output([], out_path)
        sr, data = wavfile.read(out_path)
        # 4 preset notes; ensure the rendered fallback is not silence
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertGreater(np.max(np.abs(data)), 100)

    def test_valid_melody_with_confused_emotion_keeps_melody(self):
        # A legitimately 'confused' emotion on a VALID melody must not
        # be replaced by the preset confused melody
        melody_log = [{"pitch": "C5", "duration": 0.3}]
        out_path = os.path.join(self.tmpdir, "legit_confused.wav")
        result = synthesise_output(melody_log, out_path, emotion="confused")
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), int(0.3 * SAMPLE_RATE))
        self.assertEqual(result, "confused")

    def test_preset_style_melody_renders_for_every_voice(self):
        # A preset-melody-style log (plain pitch strings + fixed
        # durations, exactly the shape the click-noise presets will
        # use) must render as a non-silent, correctly-sized 16-bit wav
        # for every character voice in both happy and angry delivery.
        preset_style_log = [
            {"pitch": "C#5", "duration": 0.3},
            {"pitch": "F5",  "duration": 0.3},
            {"pitch": "F#5", "duration": 0.3},
        ]
        expected_frames = int(0.3 * SAMPLE_RATE) * 3
        for char in VOICE_PROFILES:
            for emotion in ("happy", "angry"):
                out_path = os.path.join(
                    self.tmpdir, f"preset_{char}_{emotion}.wav")
                result = synthesise_output(
                    preset_style_log, out_path,
                    character=char, emotion=emotion)
                self.assertEqual(result, emotion)
                self.assertTrue(os.path.exists(out_path))
                sr, data = wavfile.read(out_path)
                self.assertEqual(sr, SAMPLE_RATE)
                self.assertEqual(len(data), expected_frames)
                self.assertGreater(np.max(np.abs(data)), 100)


# ---------------------------------------------------------------------
# Duration / emotion decision logic
# ---------------------------------------------------------------------
class TestDurationAndEmotion(AudioToolsTestCase):

    def test_duration_from_file_path(self):
        duration = _get_raw_audio_duration(self.one_sec_wav)
        self.assertAlmostEqual(duration, 1.0, delta=0.05)

    def test_duration_from_bytes(self):
        with open(self.one_sec_wav, "rb") as f:
            raw_bytes = f.read()
        duration = _get_raw_audio_duration(raw_bytes)
        self.assertAlmostEqual(duration, 1.0, delta=0.05)

    def test_duration_invalid_path_returns_zero(self):
        self.assertEqual(
            _get_raw_audio_duration(os.path.join(self.tmpdir, "nope.wav")),
            0.0)

    def test_empty_melody_vs_real_input_is_confused(self):
        emotion = determine_emotion([], self.one_sec_wav)
        self.assertEqual(emotion, "confused")

    def test_short_melody_vs_long_input_is_confused(self):
        melody_log = [{"pitch": "D5", "duration": 0.3}]
        emotion = determine_emotion(melody_log, self.one_sec_wav)
        self.assertEqual(emotion, "confused")

    def test_matching_melody_uses_probability_weights(self):
        melody_log = [{"pitch": "D5", "duration": 1.0}]
        emotion = determine_emotion(
            melody_log, self.one_sec_wav,
            choices=["none", "happy", "sad", "angry"],
            probabilities=[1.0, 0.0, 0.0, 0.0])
        self.assertEqual(emotion, "none")

    def test_weighted_draw_matches_np_random_choice_structure(self):
        # Regression guard: the random probability structure must stay
        # a plain np.random.choice over (choices, probabilities) so the
        # click-noise decision can reuse the identical mechanism.
        # With a pinned seed the draw must equal a manual np.random.choice.
        melody_log = [{"pitch": "D5", "duration": 1.0}]
        choices = ["happy", "angry"]
        probabilities = [0.7, 0.3]
        np.random.seed(42)
        emotion = determine_emotion(
            melody_log, self.one_sec_wav,
            choices=choices, probabilities=probabilities)
        np.random.seed(42)
        expected = str(np.random.choice(choices, p=probabilities))
        self.assertEqual(emotion, expected)
        self.assertIn(emotion, choices)


# ---------------------------------------------------------------------
# Raw duration reading across container formats (mislabeled uploads)
# ---------------------------------------------------------------------
@unittest.skipUnless(FFMPEG, "ffmpeg required to synthesise webm fixtures")
class TestRawAudioDurationFormats(AudioToolsTestCase):
    """Tests for _get_raw_audio_duration with non-WAV containers.

    The browser MediaRecorder emits webm/ogg bytes even though the
    frontend labels them "audio/wav", so the saved input file is often
    a webm file with a .wav extension. BASELINE: wave.open raises
    "file does not start with RIFF id" and the duration defaults to
    0.0, silently disabling the determine_emotion() timeline check.
    NEW behaviour: fall back to librosa so real browser uploads are
    measured correctly; only truly undecodable input yields 0.0.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.webm_as_wav = cls.make_webm("browser_upload.wav")
        with open(cls.webm_as_wav, "rb") as f:
            cls.webm_bytes = f.read()

    def test_duration_from_mislabeled_webm_path(self):
        # BASELINE: 0.0 (RIFF error). NEW: ~1.0 via the librosa fallback.
        duration = _get_raw_audio_duration(self.webm_as_wav)
        self.assertAlmostEqual(duration, 1.0, delta=0.05)

    def test_duration_from_webm_bytes(self):
        # BASELINE: 0.0. NEW: ~1.0 via the librosa fallback.
        duration = _get_raw_audio_duration(self.webm_bytes)
        self.assertAlmostEqual(duration, 1.0, delta=0.05)

    def test_wav_still_uses_fast_wave_path(self):
        # Existing behaviour must be preserved for genuine WAVs.
        duration = _get_raw_audio_duration(self.one_sec_wav)
        self.assertAlmostEqual(duration, 1.0, delta=0.05)

    def test_garbage_bytes_fail_safe_to_zero(self):
        # Neither wave nor librosa can decode random garbage; must
        # return 0.0 without raising.
        with open(self.corrupt_wav, "rb") as f:
            duration = _get_raw_audio_duration(f.read())
        self.assertEqual(duration, 0.0)

    def test_garbage_path_fail_safe_to_zero(self):
        self.assertEqual(_get_raw_audio_duration(self.corrupt_wav), 0.0)

    def test_none_input_returns_zero(self):
        self.assertEqual(_get_raw_audio_duration(None), 0.0)

    def test_missing_path_returns_zero(self):
        self.assertEqual(
            _get_raw_audio_duration(os.path.join(self.tmpdir, "nope.wav")),
            0.0)


# ---------------------------------------------------------------------
# determine_emotion() robustness and end-to-end emotion decisions
# ---------------------------------------------------------------------
@unittest.skipUnless(FFMPEG, "ffmpeg required to synthesise webm fixtures")
class TestDetermineEmotionEdgeCases(AudioToolsTestCase):
    """Edge-case and integration tests for determine_emotion()."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.webm_as_wav = cls.make_webm("browser_upload.wav")

    def test_none_melody_log_returns_confused(self):
        # BASELINE: TypeError from sum() over None. NEW: graceful
        # "confused", mirroring synthesise_output's empty-log fallback.
        self.assertEqual(determine_emotion(None, self.one_sec_wav), "confused")

    def test_missing_duration_key_treated_as_zero(self):
        # Notes without a "duration" key contribute 0.0 and must not crash.
        melody_log = [{"pitch": "D5"}]
        self.assertEqual(
            determine_emotion(melody_log, self.one_sec_wav), "confused")

    def test_none_duration_value_treated_as_zero(self):
        melody_log = [{"pitch": "D5", "duration": None}]
        self.assertEqual(
            determine_emotion(melody_log, self.one_sec_wav), "confused")

    def test_misaligned_melody_vs_mislabeled_webm_is_confused(self):
        # THE core integration bug: melody 0.3s vs ~1.0s webm upload
        # mislabeled as .wav. BASELINE: unreadable duration -> 0.0 ->
        # check silently disabled -> random "none"/"happy".
        # NEW: timeline check fires -> "confused".
        melody_log = [{"pitch": "D5", "duration": 0.3}]
        self.assertEqual(
            determine_emotion(melody_log, self.webm_as_wav), "confused")

    def test_aligned_melody_vs_mislabeled_webm_not_confused(self):
        # Melody covering >= 75% of the webm timeline must reach the
        # random draw instead of being forced to "confused".
        melody_log = [{"pitch": "D5", "duration": 1.0}]
        emotion = determine_emotion(
            melody_log, self.webm_as_wav,
            choices=["none", "happy", "sad", "angry"],
            probabilities=[1.0, 0.0, 0.0, 0.0])
        self.assertEqual(emotion, "none")

    def test_unreadable_input_with_melody_uses_random_draw(self):
        # Completely undecodable raw input: the timeline check cannot
        # run, so the weighted random draw still decides (with the
        # caveat logged). Deterministic weights pin the outcome.
        melody_log = [{"pitch": "D5", "duration": 0.3}]
        with open(self.corrupt_wav, "rb") as f:
            raw_bytes = f.read()
        emotion = determine_emotion(
            melody_log, raw_bytes,
            choices=["none", "happy", "sad", "angry"],
            probabilities=[0.0, 1.0, 0.0, 0.0])
        self.assertEqual(emotion, "happy")

    def test_empty_melody_is_confused_even_with_unreadable_input(self):
        # No melody at all is always a confused response, regardless
        # of whether the raw input can be decoded.
        self.assertEqual(
            determine_emotion([], self.corrupt_wav), "confused")

    def test_result_is_plain_str(self):
        # np.random.choice returns numpy str_; callers (app.py /
        # synthesise_output dict lookups) must receive a plain str.
        melody_log = [{"pitch": "D5", "duration": 1.0}]
        emotion = determine_emotion(
            melody_log, self.one_sec_wav,
            choices=["none", "happy"], probabilities=[1.0, 0.0])
        self.assertIsInstance(emotion, str)


# ---------------------------------------------------------------------
# Click-noise API (clickable character feature)
# ---------------------------------------------------------------------
# The click-noise API does not exist before the feature is implemented;
# the imports are guarded so the baseline suite stays runnable in the
# red phase and the new tests report as "expected failure: not
# implemented yet" instead of crashing collection.
try:
    from audio_tools import (
        HAPPY_NOISE_MELODY,
        ANGRY_NOISE_MELODY,
        CLICK_NOISE_ANGER_THRESHOLD,
        determine_noise_emotion,
        generate_click_noise,
    )
    NOISE_API_AVAILABLE = True
except ImportError:
    NOISE_API_AVAILABLE = False


NOISE_API_PENDING = "click-noise API not implemented yet"


@unittest.skipUnless(NOISE_API_AVAILABLE, NOISE_API_PENDING)
class TestClickNoiseMelodyPresets(unittest.TestCase):
    """The preset click-noise melodies and their note validity."""

    def test_happy_melody_preset_sequence(self):
        pitches = [note["pitch"] for note in HAPPY_NOISE_MELODY]
        self.assertEqual(pitches, ['C#5', 'F5', 'F#5', 'G#5', 'C#6'])

    def test_angry_melody_preset_sequence(self):
        pitches = [note["pitch"] for note in ANGRY_NOISE_MELODY]
        self.assertEqual(pitches, ['F5', 'F#5', 'F5', 'F#5'])

    def test_preset_notes_parse_and_durations_positive(self):
        for melody in (HAPPY_NOISE_MELODY, ANGRY_NOISE_MELODY):
            self.assertGreater(len(melody), 0)
            for note in melody:
                self.assertIn("pitch", note)
                self.assertIn("duration", note)
                self.assertGreater(_get_frequency(note["pitch"]), 0.0)
                self.assertGreater(float(note["duration"]), 0.0)


@unittest.skipUnless(NOISE_API_AVAILABLE, NOISE_API_PENDING)
class TestDetermineNoiseEmotion(unittest.TestCase):
    """Click-count driven happy/none draw for the clickable character."""

    def test_returns_only_happy_or_none(self):
        for click_count in range(CLICK_NOISE_ANGER_THRESHOLD):
            emotion = determine_noise_emotion(click_count)
            self.assertIn(emotion, ("happy", "none"))
            self.assertIsInstance(emotion, str)

    def test_overclick_returns_angry(self):
        for click_count in range(CLICK_NOISE_ANGER_THRESHOLD,
                                 CLICK_NOISE_ANGER_THRESHOLD + 3):
            self.assertEqual(determine_noise_emotion(click_count), "angry")

    def test_below_threshold_uses_default_probability_structure(self):
        # Same np.random.choice structure as determine_emotion's
        # default draw, restricted to the noise choices: 0.70 happy /
        # 0.30 none while the character has not been over-clicked.
        np.random.seed(7)
        expected = str(np.random.choice(["happy", "none"], p=[0.7, 0.3]))
        np.random.seed(7)
        self.assertEqual(determine_noise_emotion(0), expected)

    def test_at_and_above_threshold_forces_angry(self):
        for click_count in range(CLICK_NOISE_ANGER_THRESHOLD,
                                 CLICK_NOISE_ANGER_THRESHOLD + 3):
            for seed in range(5):
                np.random.seed(seed)
                self.assertEqual(
                    determine_noise_emotion(click_count), "angry")


@unittest.skipUnless(NOISE_API_AVAILABLE, NOISE_API_PENDING)
class TestGenerateClickNoise(AudioToolsTestCase):
    """Renders the click noise into system_noise.wav."""

    @staticmethod
    def _melody_frames(melody):
        return int(SAMPLE_RATE * sum(float(n["duration"]) for n in melody))

    def test_renders_noise_wav_of_preset_length(self):
        # Above the threshold the angry noise is forced, so the render
        # is fully deterministic in length.
        out_path = os.path.join(self.tmpdir, "system_noise.wav")
        emotion = generate_click_noise("default_cat",
                                       CLICK_NOISE_ANGER_THRESHOLD,
                                       out_path)
        self.assertEqual(emotion, "angry")
        sr, data = wavfile.read(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(len(data), self._melody_frames(ANGRY_NOISE_MELODY))
        self.assertGreater(np.max(np.abs(data)), 100)

    def test_output_length_always_matches_drawn_emotion(self):
        # Happy and angry presets have different note counts, so the
        # rendered length reveals which melody was actually used: it
        # must always match the emotion returned.
        out_path = os.path.join(self.tmpdir, "length_check.wav")
        happy_frames = self._melody_frames(HAPPY_NOISE_MELODY)
        angry_frames = self._melody_frames(ANGRY_NOISE_MELODY)
        self.assertNotEqual(happy_frames, angry_frames)
        for click_count in range(0, CLICK_NOISE_ANGER_THRESHOLD + 2):
            emotion = generate_click_noise("default_cat",
                                           click_count, out_path)
            sr, data = wavfile.read(out_path)
            expected = (happy_frames if emotion in ("happy", "none")
                        else angry_frames)
            self.assertEqual(len(data), expected)

    def test_never_touches_system_reply_wav(self):
        # system_reply.wav is reserved for direct user-to-audiopet
        # interactions: click noises must never write or overwrite it.
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        write_wav(reply_path, make_tone(440.0, 0.5))
        with open(reply_path, "rb") as f:
            reply_before = f.read()
        noise_path = os.path.join(self.tmpdir, "system_noise.wav")
        generate_click_noise("default_cat", 0, noise_path)
        self.assertTrue(os.path.exists(reply_path))
        with open(reply_path, "rb") as f:
            self.assertEqual(f.read(), reply_before)

    def test_does_not_create_system_reply_wav_if_absent(self):
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        if os.path.exists(reply_path):
            os.remove(reply_path)
        noise_path = os.path.join(self.tmpdir, "noise_only.wav")
        generate_click_noise("default_cat", 0, noise_path)
        self.assertFalse(os.path.exists(reply_path))

    def test_overwrites_previous_noise_file(self):
        # A stale/longer system_noise.wav must be fully replaced, never
        # appended to or left behind from an earlier click.
        out_path = os.path.join(self.tmpdir, "overwrite_check.wav")
        write_wav(out_path, make_tone(220.0, 5.0))
        emotion = generate_click_noise("default_cat",
                                       CLICK_NOISE_ANGER_THRESHOLD,
                                       out_path)
        self.assertEqual(emotion, "angry")
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), self._melody_frames(ANGRY_NOISE_MELODY))

    def test_unknown_character_falls_back_to_default(self):
        out_path = os.path.join(self.tmpdir, "unknown_char_noise.wav")
        emotion = generate_click_noise("not_a_character",
                                       CLICK_NOISE_ANGER_THRESHOLD, out_path)
        self.assertEqual(emotion, "angry")
        self.assertTrue(os.path.exists(out_path))

    def test_every_voice_renders_click_noise(self):
        for char in VOICE_PROFILES:
            out_path = os.path.join(self.tmpdir, f"noise_{char}.wav")
            emotion = generate_click_noise(char, 0, out_path)
            self.assertIn(emotion, ("happy", "none"))
            self.assertTrue(os.path.exists(out_path))


# ---------------------------------------------------------------------
# Short term memory feature
# ---------------------------------------------------------------------
# The short-term memory API does not exist before the feature is
# implemented; the imports are guarded so the baseline suite stays
# runnable in the red phase and the new tests report as "expected
# failure: not implemented yet" instead of crashing collection.
try:
    from audio_tools import (
        MEMORY_SIZE,
        MEMORY_SING_PROBABILITY,
        add_to_memory,
        pick_memory_melody,
        should_sing_from_memory,
    )
    MEMORY_API_AVAILABLE = True
except ImportError:
    MEMORY_API_AVAILABLE = False

# generate_click_noise() already exists (click-noise feature) but only
# gains the memory parameters in a later step; detect that support via
# inspect so the integration tests skip cleanly until it lands.
import inspect

from audio_tools import generate_click_noise as _generate_click_noise_probe

GENERATE_MEMORY_SUPPORT = "memory" in inspect.signature(
    _generate_click_noise_probe).parameters

MEMORY_API_PENDING = "short term memory API not implemented yet"
GENERATE_MEMORY_PENDING = "generate_click_noise() memory parameters not implemented yet"

# Distinct-length test melodies so rendered wav length reveals which
# melody was picked (happy/angry presets and memory logs all differ).
MEMORY_LOG_A = [
    {"pitch": "C5", "duration": 0.25},
    {"pitch": "E5", "duration": 0.25},
]
MEMORY_LOG_B = [{"pitch": "G5", "duration": 0.4}]
MEMORY_LOG_C = [
    {"pitch": "A5", "duration": 0.3},
    {"pitch": "B5", "duration": 0.3},
    {"pitch": "C6", "duration": 0.3},
]


@unittest.skipUnless(MEMORY_API_AVAILABLE, MEMORY_API_PENDING)
class TestMemoryConstants(unittest.TestCase):
    """The tunable short-term memory constants."""

    def test_memory_size_default_is_five(self):
        self.assertEqual(MEMORY_SIZE, 5)

    def test_memory_size_is_positive_int(self):
        self.assertIsInstance(MEMORY_SIZE, int)
        self.assertGreater(MEMORY_SIZE, 0)

    def test_sing_probability_is_controllable_fraction(self):
        self.assertIsInstance(MEMORY_SING_PROBABILITY, float)
        self.assertGreater(MEMORY_SING_PROBABILITY, 0.0)
        self.assertLess(MEMORY_SING_PROBABILITY, 1.0)


@unittest.skipUnless(MEMORY_API_AVAILABLE, MEMORY_API_PENDING)
class TestAddToMemory(unittest.TestCase):
    """Storage of successfully detected melodies, capped to the latest n."""

    def test_appends_non_empty_melody(self):
        memory = []
        result = add_to_memory(MEMORY_LOG_A, memory=memory)
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0], MEMORY_LOG_A)
        self.assertEqual(result, memory)

    def test_empty_melody_ignored(self):
        # Confused-noise fallback cases (empty log) must never be stored
        memory = [MEMORY_LOG_A]
        add_to_memory([], memory=memory)
        self.assertEqual(len(memory), 1)

    def test_none_melody_ignored(self):
        memory = []
        add_to_memory(None, memory=memory)
        self.assertEqual(memory, [])

    def test_non_list_melody_ignored(self):
        memory = []
        for bad in ("not a melody", 42, {"pitch": "C5", "duration": 0.3}):
            add_to_memory(bad, memory=memory)
        self.assertEqual(memory, [])

    def test_malformed_notes_ignored(self):
        # Entries must all be note dicts with pitch and duration keys
        memory = []
        for bad in ([1, 2, 3],
                    [{"pitch": "C5"}],
                    [{"duration": 0.3}],
                    [{"pitch": "C5", "duration": 0.3}, "garbage"],
                    [[]]):
            add_to_memory(bad, memory=memory)
        self.assertEqual(memory, [])

    def test_capacity_trims_to_latest_n(self):
        memory = []
        for log in (MEMORY_LOG_A, MEMORY_LOG_B, MEMORY_LOG_C):
            add_to_memory(log, memory=memory)
        # Default capacity MEMORY_SIZE=5: nothing trimmed yet
        self.assertEqual(len(memory), 3)
        extra = [{"pitch": "D6", "duration": 0.2}]
        for i in range(4):
            add_to_memory([{"pitch": "D6", "duration": 0.2 + i * 0.01}],
                          memory=memory)
        self.assertEqual(len(memory), MEMORY_SIZE)
        # 7 logs with capacity 5: the two oldest (A, B) are evicted, the
        # newest entries retained in order
        self.assertEqual(memory[0], MEMORY_LOG_C)
        self.assertEqual(memory[-1], [{"pitch": "D6", "duration": 0.23}])

    def test_custom_capacity(self):
        memory = []
        add_to_memory(MEMORY_LOG_A, capacity=2, memory=memory)
        add_to_memory(MEMORY_LOG_B, capacity=2, memory=memory)
        add_to_memory(MEMORY_LOG_C, capacity=2, memory=memory)
        self.assertEqual(len(memory), 2)
        self.assertEqual(memory[0], MEMORY_LOG_B)
        self.assertEqual(memory[1], MEMORY_LOG_C)

    def test_stores_copy_not_reference(self):
        # Later mutation of the caller's log must not corrupt memory
        memory = []
        log = [{"pitch": "C5", "duration": 0.3}]
        add_to_memory(log, memory=memory)
        log.append({"pitch": "E5", "duration": 0.3})
        self.assertEqual(len(memory[0]), 1)

    def test_module_level_memory_used_by_default(self):
        from audio_tools import short_term_memory
        snapshot = list(short_term_memory)
        try:
            add_to_memory(MEMORY_LOG_A)
            self.assertEqual(len(short_term_memory), len(snapshot) + 1)
            add_to_memory([])
            add_to_memory(None)
            self.assertEqual(len(short_term_memory), len(snapshot) + 1)
        finally:
            short_term_memory[:] = snapshot


@unittest.skipUnless(MEMORY_API_AVAILABLE, MEMORY_API_PENDING)
class TestPickMemoryMelody(unittest.TestCase):
    """Random selection of a melody to sing from short-term memory."""

    def test_empty_memory_returns_none(self):
        self.assertIsNone(pick_memory_melody([]))
        self.assertIsNone(pick_memory_melody(None))

    def test_returns_member_of_memory(self):
        memory = [MEMORY_LOG_A, MEMORY_LOG_B, MEMORY_LOG_C]
        for _ in range(10):
            self.assertIn(pick_memory_melody(memory), memory)

    def test_seeded_draw_matches_np_random_choice(self):
        # Regression guard: selection must reuse the plain np.random
        # draw structure so it stays deterministic under a pinned seed.
        memory = [MEMORY_LOG_A, MEMORY_LOG_B, MEMORY_LOG_C]
        np.random.seed(11)
        result = pick_memory_melody(memory)
        np.random.seed(11)
        expected = memory[int(np.random.randint(len(memory)))]
        self.assertEqual(result, expected)


@unittest.skipUnless(MEMORY_API_AVAILABLE, MEMORY_API_PENDING)
class TestShouldSingFromMemory(unittest.TestCase):
    """Controllable-probability gate for singing from memory on clicks."""

    def test_empty_memory_never_sings(self):
        # Even at probability 1.0 there is nothing to sing from
        for memory in ([], None):
            np.random.seed(0)
            self.assertFalse(
                should_sing_from_memory(memory, sing_probability=1.0))

    def test_probability_one_always_sings_with_memory(self):
        self.assertTrue(
            should_sing_from_memory([MEMORY_LOG_A], sing_probability=1.0))

    def test_probability_zero_never_sings_with_memory(self):
        self.assertFalse(
            should_sing_from_memory([MEMORY_LOG_A], sing_probability=0.0))

    def test_seeded_draw_matches_np_random_random(self):
        # The gate must be a plain probability draw (np.random.random()
        # < p) so it stays deterministic under a pinned seed.
        np.random.seed(3)
        expected = np.random.random() < 0.5
        np.random.seed(3)
        result = should_sing_from_memory([MEMORY_LOG_A],
                                         sing_probability=0.5)
        self.assertEqual(result, expected)

    def test_default_probability_uses_constant(self):
        np.random.seed(5)
        expected = np.random.random() < MEMORY_SING_PROBABILITY
        np.random.seed(5)
        result = should_sing_from_memory([MEMORY_LOG_A])
        self.assertEqual(result, expected)


@unittest.skipUnless(MEMORY_API_AVAILABLE, MEMORY_API_PENDING)
@unittest.skipUnless(GENERATE_MEMORY_SUPPORT, GENERATE_MEMORY_PENDING)
class TestGenerateClickNoiseMemory(AudioToolsTestCase):
    """generate_click_noise() singing a memory melody as system_noise.wav."""

    @staticmethod
    def _melody_frames(melody):
        return int(SAMPLE_RATE * sum(float(n["duration"]) for n in melody))

    def test_sings_memory_melody_at_probability_one(self):
        memory = [MEMORY_LOG_A, MEMORY_LOG_B]
        out_path = os.path.join(self.tmpdir, "memory_noise.wav")
        np.random.seed(2)
        emotion = generate_click_noise("default_cat", 0, out_path,
                                       memory=memory, sing_probability=1.0)
        # Mirror the full RNG sequence inside generate_click_noise():
        # the gate draw fires first, then the melody pick.
        np.random.seed(2)
        self.assertTrue(
            should_sing_from_memory(memory, sing_probability=1.0))
        expected_melody = pick_memory_melody(memory)
        self.assertIn(emotion, ("happy", "none"))
        self.assertIsInstance(emotion, str)
        sr, data = wavfile.read(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(len(data), self._melody_frames(expected_melody))
        self.assertGreater(np.max(np.abs(data)), 100)

    def test_single_melody_memory_always_sings_it(self):
        memory = [MEMORY_LOG_B]
        out_path = os.path.join(self.tmpdir, "single_memory.wav")
        for seed in range(5):
            np.random.seed(seed)
            emotion = generate_click_noise(
                "default_cat", 0, out_path,
                memory=list(memory), sing_probability=1.0)
            self.assertIn(emotion, ("happy", "none"))
            sr, data = wavfile.read(out_path)
            self.assertEqual(len(data), self._melody_frames(MEMORY_LOG_B))

    def test_empty_memory_falls_back_to_preset_noise(self):
        # Memory is empty: the click must behave exactly like the
        # baseline feature (preset happy/angry noise), never a crash.
        out_path = os.path.join(self.tmpdir, "empty_memory.wav")
        happy_frames = self._melody_frames(HAPPY_NOISE_MELODY)
        angry_frames = self._melody_frames(ANGRY_NOISE_MELODY)
        for click_count in range(0, CLICK_NOISE_ANGER_THRESHOLD + 2):
            np.random.seed(click_count)
            emotion = generate_click_noise(
                "default_cat", click_count, out_path,
                memory=[], sing_probability=1.0)
            sr, data = wavfile.read(out_path)
            expected = (happy_frames if emotion in ("happy", "none")
                        else angry_frames)
            self.assertEqual(len(data), expected)

    def test_probability_zero_uses_preset_noise(self):
        memory = [MEMORY_LOG_A, MEMORY_LOG_C]
        out_path = os.path.join(self.tmpdir, "prob_zero.wav")
        happy_frames = self._melody_frames(HAPPY_NOISE_MELODY)
        angry_frames = self._melody_frames(ANGRY_NOISE_MELODY)
        for click_count in range(0, CLICK_NOISE_ANGER_THRESHOLD + 2):
            emotion = generate_click_noise(
                "default_cat", click_count, out_path,
                memory=memory, sing_probability=0.0)
            sr, data = wavfile.read(out_path)
            expected = (happy_frames if emotion in ("happy", "none")
                        else angry_frames)
            self.assertEqual(len(data), expected)

    def test_default_probability_when_sing_probability_omitted(self):
        # Omitting sing_probability must use MEMORY_SING_PROBABILITY as
        # the gate: with a pinned seed the result must match a manual
        # should_sing_from_memory() draw.
        memory = [MEMORY_LOG_A]
        out_path = os.path.join(self.tmpdir, "default_prob.wav")
        np.random.seed(9)
        expected_sing = should_sing_from_memory(memory)
        np.random.seed(9)
        emotion = generate_click_noise("default_cat", 0, out_path,
                                       memory=memory)
        sr, data = wavfile.read(out_path)
        if expected_sing:
            self.assertEqual(len(data), self._melody_frames(MEMORY_LOG_A))
            self.assertIn(emotion, ("happy", "none"))
        else:
            preset_frames = (self._melody_frames(HAPPY_NOISE_MELODY)
                             if emotion in ("happy", "none")
                             else self._melody_frames(ANGRY_NOISE_MELODY))
            self.assertEqual(len(data), preset_frames)

    def test_overclick_never_sings_from_memory(self):
        # Once the click threshold is reached the pet is fed up: the
        # memory is never used, only the preset angry noise plays,
        # until the click count resets (e.g. after a new recording).
        # The angry preset has a distinct frame length from every
        # memory melody, so the rendered length proves which was used.
        memory = [MEMORY_LOG_A, MEMORY_LOG_B]
        out_path = os.path.join(self.tmpdir, "memory_overclick.wav")
        angry_frames = self._melody_frames(ANGRY_NOISE_MELODY)
        self.assertNotIn(angry_frames,
                         [self._melody_frames(m) for m in memory])
        for click_count in range(CLICK_NOISE_ANGER_THRESHOLD,
                                 CLICK_NOISE_ANGER_THRESHOLD + 3):
            for seed in range(3):
                np.random.seed(seed)
                emotion = generate_click_noise(
                    "default_cat", click_count, out_path,
                    memory=list(memory), sing_probability=1.0)
                self.assertEqual(emotion, "angry")
                sr, data = wavfile.read(out_path)
                self.assertEqual(len(data), angry_frames)

    def test_memory_sing_never_touches_system_reply_wav(self):
        # system_reply.wav stays reserved for direct user-to-audiopet
        # interactions, exactly as for the baseline click noise.
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        write_wav(reply_path, make_tone(440.0, 0.5))
        with open(reply_path, "rb") as f:
            reply_before = f.read()
        noise_path = os.path.join(self.tmpdir, "memory_reply_guard.wav")
        generate_click_noise("default_cat", 0, noise_path,
                             memory=[MEMORY_LOG_A], sing_probability=1.0)
        self.assertTrue(os.path.exists(reply_path))
        with open(reply_path, "rb") as f:
            self.assertEqual(f.read(), reply_before)

    def test_memory_sing_does_not_create_system_reply_wav(self):
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        if os.path.exists(reply_path):
            os.remove(reply_path)
        noise_path = os.path.join(self.tmpdir, "memory_noise_only.wav")
        generate_click_noise("default_cat", 0, noise_path,
                             memory=[MEMORY_LOG_A], sing_probability=1.0)
        self.assertFalse(os.path.exists(reply_path))

    def test_memory_mutation_after_click_does_not_corrupt_render(self):
        # The stored copy must stay independent of the caller's list,
        # so rendering cannot be affected by later edits to the log.
        log = [{"pitch": "C5", "duration": 0.3}]
        memory = []
        add_to_memory(log, memory=memory)
        log[0]["duration"] = 9.9
        out_path = os.path.join(self.tmpdir, "copy_guard.wav")
        generate_click_noise("default_cat", 0, out_path,
                             memory=memory, sing_probability=1.0)
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), self._melody_frames([{"pitch": "C5",
                                                          "duration": 0.3}]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
