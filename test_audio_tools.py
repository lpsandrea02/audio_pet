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
    CONFUSED_MELODY,
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


# ---------------------------------------------------------------------
# Idle character noises feature
# ---------------------------------------------------------------------
# The idle-noise API does not exist before the feature is implemented;
# the imports are guarded so the baseline suite stays runnable in the
# red phase and the new tests report as "expected failure: not
# implemented yet" instead of crashing collection.
try:
    from audio_tools import (
        IDLE_NOISE_MIN_DELAY,
        IDLE_NOISE_MAX_DELAY,
        IDLE_NOISE_MAX_COUNT,
        SAD_NOISE_MELODY,
        get_idle_noise_delay,
        pick_idle_noise,
        should_emit_idle_noise,
        generate_idle_noise,
    )
    IDLE_API_AVAILABLE = True
except ImportError:
    IDLE_API_AVAILABLE = False


IDLE_API_PENDING = "idle-noise API not implemented yet"

# Distinct-length memory logs so rendered wav length reveals which
# melody was used (they also differ from every preset melody length).
IDLE_MEMORY_LOG_A = [{"pitch": "E6", "duration": 0.35}]
IDLE_MEMORY_LOG_B = [
    {"pitch": "D6", "duration": 0.2},
    {"pitch": "C6", "duration": 0.25},
]


@unittest.skipUnless(IDLE_API_AVAILABLE, IDLE_API_PENDING)
class TestIdleNoiseConstants(unittest.TestCase):
    """The tunable idle-noise constants and the sad preset melody."""

    def test_delay_range_is_valid(self):
        self.assertIsInstance(IDLE_NOISE_MIN_DELAY, (int, float))
        self.assertIsInstance(IDLE_NOISE_MAX_DELAY, (int, float))
        self.assertGreater(IDLE_NOISE_MIN_DELAY, 0)
        self.assertGreater(IDLE_NOISE_MAX_DELAY, IDLE_NOISE_MIN_DELAY)

    def test_max_count_is_positive_int(self):
        self.assertIsInstance(IDLE_NOISE_MAX_COUNT, int)
        self.assertGreater(IDLE_NOISE_MAX_COUNT, 0)

    def test_sad_melody_preset_sequence(self):
        pitches = [note["pitch"] for note in SAD_NOISE_MELODY]
        self.assertEqual(pitches, ['B5', 'A#5', 'A5', 'G#5', 'G5'])

    def test_sad_melody_notes_parse_and_durations_positive(self):
        self.assertGreater(len(SAD_NOISE_MELODY), 0)
        for note in SAD_NOISE_MELODY:
            self.assertIn("pitch", note)
            self.assertIn("duration", note)
            self.assertGreater(_get_frequency(note["pitch"]), 0.0)
            self.assertGreater(float(note["duration"]), 0.0)

    def test_sad_melody_distinct_length_from_other_presets(self):
        # The sad noise's total duration must differ from every other
        # preset melody so a rendered wav length proves which was used.
        def frames(melody):
            return int(SAMPLE_RATE * sum(float(n["duration"]) for n in melody))
        sad_frames = frames(SAD_NOISE_MELODY)
        for preset in (HAPPY_NOISE_MELODY, ANGRY_NOISE_MELODY,
                       CONFUSED_MELODY):
            self.assertNotEqual(frames(preset), sad_frames)


@unittest.skipUnless(IDLE_API_AVAILABLE, IDLE_API_PENDING)
class TestGetIdleNoiseDelay(unittest.TestCase):
    """Random idle-noise spacing between the min/max delay constants."""

    def test_delay_within_default_range(self):
        for _ in range(20):
            delay = get_idle_noise_delay()
            self.assertGreaterEqual(delay, IDLE_NOISE_MIN_DELAY)
            self.assertLessEqual(delay, IDLE_NOISE_MAX_DELAY)

    def test_delay_is_number(self):
        self.assertIsInstance(get_idle_noise_delay(), (int, float))

    def test_seeded_draw_is_deterministic(self):
        np.random.seed(13)
        first = get_idle_noise_delay()
        np.random.seed(13)
        second = get_idle_noise_delay()
        self.assertEqual(first, second)

    def test_custom_range_respected(self):
        for _ in range(20):
            delay = get_idle_noise_delay(min_delay=2.0, max_delay=5.0)
            self.assertGreaterEqual(delay, 2.0)
            self.assertLessEqual(delay, 5.0)

    def test_inverted_range_does_not_raise(self):
        # Fail-safe: an inverted or zero-width range must never crash
        # the caller; it should return a usable positive delay.
        delay = get_idle_noise_delay(min_delay=30.0, max_delay=10.0)
        self.assertIsInstance(delay, (int, float))
        self.assertGreater(delay, 0)


@unittest.skipUnless(IDLE_API_AVAILABLE, IDLE_API_PENDING)
class TestPickIdleNoise(AudioToolsTestCase):
    """Random selection among the idle noise options."""

    def setUp(self):
        self.PRESETS = (HAPPY_NOISE_MELODY, CONFUSED_MELODY,
                        ANGRY_NOISE_MELODY)

    def test_returns_melody_and_emotion_pair(self):
        melody, emotion = pick_idle_noise()
        self.assertIsInstance(melody, list)
        self.assertGreater(len(melody), 0)
        self.assertIsInstance(emotion, str)
        self.assertIn(emotion, ("happy", "none", "confused", "angry"))

    def test_without_memory_uses_preset_melodies(self):
        for _ in range(20):
            melody, emotion = pick_idle_noise(memory=None)
            self.assertIn(melody, self.PRESETS)

    def test_memory_probability_zero_uses_preset_melodies(self):
        memory = [IDLE_MEMORY_LOG_A, IDLE_MEMORY_LOG_B]
        for _ in range(20):
            melody, emotion = pick_idle_noise(
                memory=memory, memory_probability=0.0)
            self.assertIn(melody, self.PRESETS)

    def test_memory_probability_one_uses_memory_melody(self):
        memory = [IDLE_MEMORY_LOG_A, IDLE_MEMORY_LOG_B]
        for _ in range(20):
            melody, emotion = pick_idle_noise(
                memory=memory, memory_probability=1.0)
            self.assertIn(melody, memory)

    def test_empty_memory_falls_back_to_presets(self):
        # Even at probability 1.0 there is nothing to recall from
        memory = []
        for _ in range(10):
            melody, emotion = pick_idle_noise(
                memory=memory, memory_probability=1.0)
            self.assertIn(melody, self.PRESETS)

    def test_seeded_draw_is_deterministic(self):
        memory = [IDLE_MEMORY_LOG_A]
        np.random.seed(17)
        first = pick_idle_noise(memory=memory)
        np.random.seed(17)
        second = pick_idle_noise(memory=memory)
        self.assertEqual(first, second)

    def test_all_drawn_melodies_render_audibly(self):
        # Every melody the picker can return must be renderable by the
        # existing synthesise_output pipeline for every character voice.
        melody, emotion = pick_idle_noise()
        for char in VOICE_PROFILES:
            out_path = os.path.join(self.tmpdir, f"idle_{char}.wav")
            result = synthesise_output(melody, out_path,
                                       character=char, emotion=emotion)
            self.assertEqual(result, emotion)
            self.assertTrue(os.path.exists(out_path))
            sr, data = wavfile.read(out_path)
            self.assertGreater(np.max(np.abs(data)), 100)


@unittest.skipUnless(IDLE_API_AVAILABLE, IDLE_API_PENDING)
class TestShouldEmitIdleNoise(unittest.TestCase):
    """Gate that stops idle noises after the sad noise has played."""

    def test_emits_below_max_count(self):
        for idle_count in range(IDLE_NOISE_MAX_COUNT):
            self.assertTrue(should_emit_idle_noise(idle_count))

    def test_stops_at_max_count(self):
        self.assertFalse(should_emit_idle_noise(IDLE_NOISE_MAX_COUNT))

    def test_stays_stopped_beyond_max_count(self):
        # After the sad noise no further idle noises are generated
        for idle_count in range(IDLE_NOISE_MAX_COUNT,
                                IDLE_NOISE_MAX_COUNT + 5):
            self.assertFalse(should_emit_idle_noise(idle_count))

    def test_seeded_draw_is_deterministic(self):
        np.random.seed(19)
        first = should_emit_idle_noise(0)
        np.random.seed(19)
        second = should_emit_idle_noise(0)
        self.assertEqual(first, second)


@unittest.skipUnless(IDLE_API_AVAILABLE, IDLE_API_PENDING)
class TestGenerateIdleNoise(AudioToolsTestCase):
    """Renders the idle noise into a dedicated wav file."""

    @staticmethod
    def _melody_frames(melody):
        return int(SAMPLE_RATE * sum(float(n["duration"]) for n in melody))

    def test_sad_noise_at_max_count(self):
        # Once the fixed number of idle noises is reached the character
        # emits the sad noise with a sad emotion — and this is final.
        out_path = os.path.join(self.tmpdir, "idle_sad.wav")
        emotion, is_final = generate_idle_noise(
            "default_cat", IDLE_NOISE_MAX_COUNT, out_path)
        self.assertEqual(emotion, "sad")
        self.assertTrue(is_final)
        sr, data = wavfile.read(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(len(data), self._melody_frames(SAD_NOISE_MELODY))
        self.assertGreater(np.max(np.abs(data)), 100)

    def test_sad_noise_beyond_max_count(self):
        out_path = os.path.join(self.tmpdir, "idle_sad_beyond.wav")
        emotion, is_final = generate_idle_noise(
            "default_cat", IDLE_NOISE_MAX_COUNT + 3, out_path)
        self.assertEqual(emotion, "sad")
        self.assertTrue(is_final)
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), self._melody_frames(SAD_NOISE_MELODY))

    def test_below_max_count_renders_valid_noise(self):
        out_path = os.path.join(self.tmpdir, "idle_normal.wav")
        for idle_count in range(IDLE_NOISE_MAX_COUNT):
            np.random.seed(idle_count)
            emotion, is_final = generate_idle_noise(
                "default_cat", idle_count, out_path)
            self.assertIn(emotion, ("happy", "none", "confused", "angry"))
            self.assertIsInstance(emotion, str)
            self.assertFalse(is_final)
            self.assertTrue(os.path.exists(out_path))
            sr, data = wavfile.read(out_path)
            self.assertGreater(np.max(np.abs(data)), 100)

    def test_memory_used_at_probability_one_below_max_count(self):
        memory = [IDLE_MEMORY_LOG_A, IDLE_MEMORY_LOG_B]
        out_path = os.path.join(self.tmpdir, "idle_memory.wav")
        emotion, is_final = generate_idle_noise(
            "default_cat", 0, out_path,
            memory=memory, memory_probability=1.0)
        self.assertFalse(is_final)
        self.assertIn(emotion, ("happy", "none"))
        sr, data = wavfile.read(out_path)
        self.assertIn(len(data),
                      [self._melody_frames(m) for m in memory])

    def test_sad_noise_never_touches_system_reply_wav(self):
        # system_reply.wav is reserved for direct user-to-audiopet
        # interactions: idle noises must never write or overwrite it.
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        write_wav(reply_path, make_tone(440.0, 0.5))
        with open(reply_path, "rb") as f:
            reply_before = f.read()
        idle_path = os.path.join(self.tmpdir, "system_idle.wav")
        generate_idle_noise("default_cat", IDLE_NOISE_MAX_COUNT, idle_path)
        self.assertTrue(os.path.exists(reply_path))
        with open(reply_path, "rb") as f:
            self.assertEqual(f.read(), reply_before)

    def test_sad_noise_does_not_create_system_reply_wav_if_absent(self):
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        if os.path.exists(reply_path):
            os.remove(reply_path)
        idle_path = os.path.join(self.tmpdir, "system_idle_only.wav")
        generate_idle_noise("default_cat", IDLE_NOISE_MAX_COUNT, idle_path)
        self.assertFalse(os.path.exists(reply_path))

    def test_unknown_character_falls_back_to_default(self):
        out_path = os.path.join(self.tmpdir, "idle_unknown_char.wav")
        emotion, is_final = generate_idle_noise(
            "not_a_character", IDLE_NOISE_MAX_COUNT, out_path)
        self.assertEqual(emotion, "sad")
        self.assertTrue(is_final)
        self.assertTrue(os.path.exists(out_path))

    def test_every_voice_renders_idle_noise(self):
        for char in VOICE_PROFILES:
            out_path = os.path.join(self.tmpdir, f"idle_{char}.wav")
            emotion, is_final = generate_idle_noise(
                char, IDLE_NOISE_MAX_COUNT, out_path)
            self.assertEqual(emotion, "sad")
            self.assertTrue(is_final)
            self.assertTrue(os.path.exists(out_path))
            sr, data = wavfile.read(out_path)
            self.assertGreater(np.max(np.abs(data)), 100)

    def test_overwrites_previous_idle_file(self):
        # A stale/longer idle wav must be fully replaced, never
        # appended to or left behind from an earlier idle noise.
        out_path = os.path.join(self.tmpdir, "idle_overwrite.wav")
        write_wav(out_path, make_tone(220.0, 20.0))
        emotion, is_final = generate_idle_noise(
            "default_cat", IDLE_NOISE_MAX_COUNT, out_path)
        self.assertEqual(emotion, "sad")
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), self._melody_frames(SAD_NOISE_MELODY))


# ---------------------------------------------------------------------
# Interactive mode feature
# ---------------------------------------------------------------------
# The interactive-mode API does not exist before the feature is
# implemented; the imports are guarded so the baseline suite stays
# runnable in the red phase and the new tests report as "expected
# failure: not implemented yet" instead of crashing collection.
try:
    from audio_tools import (
        INTERACTIVE_SILENCE_THRESHOLD,
        INTERACTIVE_INTERRUPT_ANGER_THRESHOLD,
        is_user_turn_ended,
        should_interrupt_angry,
        extend_melody_log,
        generate_interrupt_reply,
        generate_interrupt_angry_noise,
    )
    INTERACTIVE_API_AVAILABLE = True
except ImportError:
    INTERACTIVE_API_AVAILABLE = False


INTERACTIVE_API_PENDING = "interactive-mode API not implemented yet"

# Distinct-length melody logs so a rendered wav length reveals which
# melody was used (they differ from every preset melody length).
INTERACTIVE_LOG_A = [
    {"pitch": "C5", "duration": 0.3},
    {"pitch": "D5", "duration": 0.3},
]
INTERACTIVE_LOG_B = [{"pitch": "E5", "duration": 0.45}]


@unittest.skipUnless(INTERACTIVE_API_AVAILABLE, INTERACTIVE_API_PENDING)
class TestInteractiveConstants(unittest.TestCase):
    """The tunable interactive-mode constants."""

    def test_silence_threshold_is_positive_number(self):
        self.assertIsInstance(INTERACTIVE_SILENCE_THRESHOLD, (int, float))
        self.assertGreater(INTERACTIVE_SILENCE_THRESHOLD, 0)

    def test_anger_threshold_is_positive_int(self):
        self.assertIsInstance(INTERACTIVE_INTERRUPT_ANGER_THRESHOLD, int)
        self.assertGreater(INTERACTIVE_INTERRUPT_ANGER_THRESHOLD, 0)


@unittest.skipUnless(INTERACTIVE_API_AVAILABLE, INTERACTIVE_API_PENDING)
class TestIsUserTurnEnded(unittest.TestCase):
    """Silence-threshold gate that ends the user's turn while streaming."""

    def test_silence_below_threshold_not_ended(self):
        self.assertFalse(is_user_turn_ended(0.0))
        self.assertFalse(is_user_turn_ended(INTERACTIVE_SILENCE_THRESHOLD / 2))

    def test_silence_at_threshold_is_ended(self):
        self.assertTrue(is_user_turn_ended(INTERACTIVE_SILENCE_THRESHOLD))

    def test_silence_above_threshold_is_ended(self):
        self.assertTrue(is_user_turn_ended(INTERACTIVE_SILENCE_THRESHOLD * 3))

    def test_custom_threshold_respected(self):
        self.assertFalse(is_user_turn_ended(0.5, threshold=1.0))
        self.assertTrue(is_user_turn_ended(1.0, threshold=1.0))
        self.assertTrue(is_user_turn_ended(2.0, threshold=1.0))

    def test_none_silence_fail_safe_false(self):
        # An unreadable silence value must never crash the streaming
        # loop; the turn simply stays active.
        self.assertFalse(is_user_turn_ended(None))

    def test_negative_silence_fail_safe_false(self):
        self.assertFalse(is_user_turn_ended(-1.0))

    def test_non_numeric_silence_fail_safe_false(self):
        for bad in ("1.0", [1.0], {"s": 1.0}):
            self.assertFalse(is_user_turn_ended(bad))


@unittest.skipUnless(INTERACTIVE_API_AVAILABLE, INTERACTIVE_API_PENDING)
class TestShouldInterruptAngry(unittest.TestCase):
    """Interruption counting: too many interrupts force the angry noise."""

    def test_below_threshold_not_angry(self):
        for count in range(INTERACTIVE_INTERRUPT_ANGER_THRESHOLD):
            self.assertFalse(should_interrupt_angry(count))

    def test_at_threshold_is_angry(self):
        self.assertTrue(
            should_interrupt_angry(INTERACTIVE_INTERRUPT_ANGER_THRESHOLD))

    def test_above_threshold_is_angry(self):
        self.assertTrue(
            should_interrupt_angry(INTERACTIVE_INTERRUPT_ANGER_THRESHOLD + 5))

    def test_custom_threshold_respected(self):
        self.assertFalse(should_interrupt_angry(1, threshold=2))
        self.assertTrue(should_interrupt_angry(2, threshold=2))

    def test_none_count_fail_safe_false(self):
        self.assertFalse(should_interrupt_angry(None))

    def test_negative_count_fail_safe_false(self):
        self.assertFalse(should_interrupt_angry(-1))

    def test_non_numeric_count_fail_safe_false(self):
        for bad in ("3", [3], {"n": 3}):
            self.assertFalse(should_interrupt_angry(bad))


@unittest.skipUnless(INTERACTIVE_API_AVAILABLE, INTERACTIVE_API_PENDING)
class TestExtendMelodyLog(unittest.TestCase):
    """Merging chunk-detected notes into the running turn melody."""

    def test_merges_new_notes(self):
        running = [{"pitch": "C5", "duration": 0.3}]
        result = extend_melody_log(running,
                                   [{"pitch": "E5", "duration": 0.2}])
        self.assertEqual(result, [
            {"pitch": "C5", "duration": 0.3},
            {"pitch": "E5", "duration": 0.2},
        ])

    def test_extends_empty_running_log(self):
        result = extend_melody_log([], INTERACTIVE_LOG_A)
        self.assertEqual(result, INTERACTIVE_LOG_A)

    def test_empty_new_notes_unchanged(self):
        running = [{"pitch": "C5", "duration": 0.3}]
        self.assertEqual(extend_melody_log(running, []), running)

    def test_none_new_notes_unchanged(self):
        running = [{"pitch": "C5", "duration": 0.3}]
        self.assertEqual(extend_melody_log(running, None), running)

    def test_malformed_new_notes_rejected(self):
        # Entries must all be note dicts with pitch and duration keys
        running = [{"pitch": "C5", "duration": 0.3}]
        for bad in ("not notes", 42,
                    [{"pitch": "E5"}],
                    [{"duration": 0.2}],
                    [{"pitch": "E5", "duration": 0.2}, "garbage"]):
            self.assertEqual(extend_melody_log(running, bad), running)

    def test_stores_copy_not_reference(self):
        # Later mutation of either the running log or the chunk notes
        # must not corrupt the merged result.
        running = [{"pitch": "C5", "duration": 0.3}]
        new_notes = [{"pitch": "E5", "duration": 0.2}]
        result = extend_melody_log(running, new_notes)
        running[0]["duration"] = 9.9
        new_notes[0]["pitch"] = "X9"
        self.assertEqual(result, [
            {"pitch": "C5", "duration": 0.3},
            {"pitch": "E5", "duration": 0.2},
        ])


@unittest.skipUnless(INTERACTIVE_API_AVAILABLE, INTERACTIVE_API_PENDING)
class TestGenerateInterruptReply(AudioToolsTestCase):
    """Synthesises the accumulated turn melody as the interactive reply."""

    @staticmethod
    def _melody_frames(melody):
        return int(SAMPLE_RATE * sum(float(n["duration"]) for n in melody))

    def test_renders_turn_melody_wav(self):
        out_path = os.path.join(self.tmpdir, "interactive_reply.wav")
        emotion = generate_interrupt_reply(INTERACTIVE_LOG_A, out_path)
        self.assertIn(emotion, ("happy", "none"))
        self.assertIsInstance(emotion, str)
        self.assertTrue(os.path.exists(out_path))
        sr, data = wavfile.read(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(len(data), self._melody_frames(INTERACTIVE_LOG_A))
        self.assertGreater(np.max(np.abs(data)), 100)

    def test_empty_melody_falls_back_to_confused_noise(self):
        # Confused-noise fallback: empty accumulated melody renders the
        # preset confused noise (same as synthesise_output's fallback).
        out_path = os.path.join(self.tmpdir, "interactive_confused.wav")
        emotion = generate_interrupt_reply([], out_path)
        self.assertEqual(emotion, "confused")
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), self._melody_frames(CONFUSED_MELODY))

    def test_none_melody_falls_back_to_confused_noise(self):
        out_path = os.path.join(self.tmpdir, "interactive_confused_none.wav")
        emotion = generate_interrupt_reply(None, out_path)
        self.assertEqual(emotion, "confused")
        self.assertTrue(os.path.exists(out_path))

    def test_never_touches_system_reply_wav(self):
        # system_reply.wav is reserved for the standard recording
        # pipeline: interactive replies must never write or overwrite
        # it (they use their own dedicated file instead).
        reply_path = os.path.join(self.tmpdir, "system_reply.wav")
        write_wav(reply_path, make_tone(440.0, 0.5))
        with open(reply_path, "rb") as f:
            reply_before = f.read()
        interactive_path = os.path.join(self.tmpdir, "system_interactive.wav")
        generate_interrupt_reply(INTERACTIVE_LOG_A, interactive_path)
        self.assertTrue(os.path.exists(reply_path))
        with open(reply_path, "rb") as f:
            self.assertEqual(f.read(), reply_before)

    def test_unknown_character_falls_back_to_default(self):
        out_path = os.path.join(self.tmpdir, "interactive_unknown.wav")
        emotion = generate_interrupt_reply(INTERACTIVE_LOG_B, out_path,
                                           character="not_a_character")
        self.assertIn(emotion, ("happy", "none"))
        sr, data = wavfile.read(out_path)
        self.assertEqual(len(data), self._melody_frames(INTERACTIVE_LOG_B))

    def test_every_voice_renders_interrupt_reply(self):
        for char in VOICE_PROFILES:
            out_path = os.path.join(self.tmpdir, f"interactive_{char}.wav")
            emotion = generate_interrupt_reply(INTERACTIVE_LOG_B, out_path,
                                               character=char)
            self.assertIn(emotion, ("happy", "none"))
            self.assertTrue(os.path.exists(out_path))


@unittest.skipUnless(INTERACTIVE_API_AVAILABLE, INTERACTIVE_API_PENDING)
class TestGenerateInterruptAngryNoise(AudioToolsTestCase):
    """Renders the forced angry noise after too many interruptions."""

    @staticmethod
    def _melody_frames(melody):
        return int(SAMPLE_RATE * sum(float(n["duration"]) for n in melody))

    def test_renders_angry_preset_melody(self):
        out_path = os.path.join(self.tmpdir, "interrupt_angry.wav")
        emotion = generate_interrupt_angry_noise("default_cat", out_path)
        self.assertEqual(emotion, "angry")
        self.assertTrue(os.path.exists(out_path))
        sr, data = wavfile.read(out_path)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(len(data), self._melody_frames(ANGRY_NOISE_MELODY))
        self.assertGreater(np.max(np.abs(data)), 100)

    def test_distinct_from_turn_melody_length(self):
        # The angry noise must not be confusable with a normal
        # interactive reply: its frame length differs from the test
        # melodies' rendered lengths.
        angry_frames = self._melody_frames(ANGRY_NOISE_MELODY)
        for log in (INTERACTIVE_LOG_A, INTERACTIVE_LOG_B):
            self.assertNotEqual(self._melody_frames(log), angry_frames)

    def test_unknown_character_falls_back_to_default(self):
        out_path = os.path.join(self.tmpdir, "interrupt_angry_unknown.wav")
        emotion = generate_interrupt_angry_noise("not_a_character", out_path)
        self.assertEqual(emotion, "angry")
        self.assertTrue(os.path.exists(out_path))


# ---------------------------------------------------------------------
# Voice-activity detection gate for interactive mode
# ---------------------------------------------------------------------
# The VAD gate does not exist before the feature is implemented; the
# imports are guarded so the baseline suite stays runnable in the red
# phase and the new tests report as "expected failure: not implemented
# yet" instead of crashing collection.
try:
    from audio_tools import is_chunk_voiced
    VAD_API_AVAILABLE = True
except ImportError:
    VAD_API_AVAILABLE = False


VAD_API_PENDING = "voice-activity gate not implemented yet"


def make_hum_tone(freq, duration, amp=0.5, sample_rate=SAMPLE_RATE):
    """Return a harmonic-rich hum/sing tone (speech-like for VAD)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 6 * t)
    return amp * (
        np.sin(2 * np.pi * freq * vibrato * t)
        + 0.5 * np.sin(2 * np.pi * freq * 2 * vibrato * t)
        + 0.25 * np.sin(2 * np.pi * freq * 3 * vibrato * t)) / 1.75


@unittest.skipUnless(VAD_API_AVAILABLE, VAD_API_PENDING)
class TestChunkVoiceDetection(AudioToolsTestCase):
    """The WebRTC VAD gate for streamed interactive chunks."""

    def test_voiced_hum_chunk_is_voiced(self):
        # A harmonic hum/sing tone chunk must be classified as voiced
        path = os.path.join(self.tmpdir, "vad_hum.wav")
        write_wav(path, make_hum_tone(523.25, 0.6))
        self.assertTrue(is_chunk_voiced(path))

    def test_silent_chunk_not_voiced(self):
        path = os.path.join(self.tmpdir, "vad_silent.wav")
        write_wav(path, np.zeros(SAMPLE_RATE // 2))
        self.assertFalse(is_chunk_voiced(path))

    def test_mostly_silent_chunk_rejected_at_strict_ratio(self):
        # 80% silence followed by a short tone: a strict custom ratio
        # must reject the chunk (silence keeps accumulating), while the
        # default lenient ratio accepts it.
        samples = np.concatenate([
            np.zeros(int(SAMPLE_RATE * 0.8)),
            make_hum_tone(523.25, 0.2),
        ])
        lenient_path = os.path.join(self.tmpdir, "vad_mixed_lenient.wav")
        strict_path = os.path.join(self.tmpdir, "vad_mixed_strict.wav")
        write_wav(lenient_path, samples)
        write_wav(strict_path, samples)
        self.assertTrue(is_chunk_voiced(lenient_path))
        self.assertFalse(is_chunk_voiced(strict_path, voiced_ratio=0.5))

    def test_corrupt_file_fail_safe_not_voiced(self):
        # Undecodable input must never crash the streaming loop
        self.assertFalse(is_chunk_voiced(self.corrupt_wav))

    def test_missing_file_fail_safe_not_voiced(self):
        self.assertFalse(
            is_chunk_voiced(os.path.join(self.tmpdir, "nope.wav")))

    def test_header_only_file_fail_safe_not_voiced(self):
        self.assertFalse(is_chunk_voiced(self.header_only_wav))


# ---------------------------------------------------------------------
# Baseline: existing chunk pipeline functions under streaming use
# ---------------------------------------------------------------------
class TestChunkPipelineBaseline(AudioToolsTestCase):
    """Baseline for the functions the streaming loop will reuse.

    Interactive mode feeds short chunk-sized wav files through
    normalize_audio() + detect_melody() continuously; these tests pin
    that the EXISTING functions already handle such short inputs
    (they must stay green before and after any modification).
    """

    def test_detect_melody_on_chunk_sized_input(self):
        # A ~0.6s chunk (typical streaming slice) of a held C5 note
        chunk_path = os.path.join(self.tmpdir, "chunk_c5.wav")
        write_wav(chunk_path, make_tone(523.25, 0.6))
        melody = detect_melody(chunk_path)
        self.assertGreater(len(melody), 0)
        for note in melody:
            self.assertIn("pitch", note)
            self.assertIn("duration", note)
            self.assertGreater(note["duration"], 0)

    def test_detect_melody_on_tiny_chunk_returns_empty(self):
        # Chunks shorter than the stability window yield no notes
        chunk_path = os.path.join(self.tmpdir, "chunk_tiny.wav")
        write_wav(chunk_path, make_tone(523.25, 0.05))
        self.assertEqual(detect_melody(chunk_path), [])

    def test_normalize_and_detect_chunk_integration(self):
        # The exact streaming sequence: normalise a quiet chunk, then
        # detect melody from the boosted output.
        chunk_in = os.path.join(self.tmpdir, "chunk_quiet.wav")
        chunk_boosted = os.path.join(self.tmpdir, "chunk_quiet_boosted.wav")
        write_wav(chunk_in, make_tone(523.25, 0.6, amp=0.01))
        normalize_audio(chunk_in, chunk_boosted)
        self.assertTrue(os.path.exists(chunk_boosted))
        melody = detect_melody(chunk_boosted)
        self.assertGreater(len(melody), 0)

    def test_normalize_chunk_silent_input_writes_nothing(self):
        chunk_in = os.path.join(self.tmpdir, "chunk_silent.wav")
        chunk_out = os.path.join(self.tmpdir, "chunk_silent_boosted.wav")
        write_wav(chunk_in, np.zeros(SAMPLE_RATE // 4))
        normalize_audio(chunk_in, chunk_out)
        self.assertFalse(os.path.exists(chunk_out))

    def test_normalize_chunk_undecodable_input_writes_nothing(self):
        chunk_in = os.path.join(self.tmpdir, "chunk_corrupt.wav")
        chunk_out = os.path.join(self.tmpdir, "chunk_corrupt_boosted.wav")
        with open(chunk_in, "wb") as f:
            f.write(os.urandom(1024))
        normalize_audio(chunk_in, chunk_out)
        self.assertFalse(os.path.exists(chunk_out))


# ---------------------------------------------------------------------
# Baseline: existing preset melodies under the sad emotion delivery
# ---------------------------------------------------------------------
class TestPresetMelodiesUnderSadEmotion(AudioToolsTestCase):
    """All preset melodies must render under the new 'sad' delivery."""

    def test_presets_render_with_sad_emotion(self):
        for preset in (HAPPY_NOISE_MELODY, ANGRY_NOISE_MELODY,
                       CONFUSED_MELODY):
            out_path = os.path.join(
                self.tmpdir, f"sad_{len(preset)}_{preset[0]['pitch']}.wav")
            result = synthesise_output(preset, out_path,
                                       character="default_cat", emotion="sad")
            self.assertEqual(result, "sad")
            self.assertTrue(os.path.exists(out_path))
            sr, data = wavfile.read(out_path)
            expected_frames = int(
                SAMPLE_RATE * sum(float(n["duration"]) for n in preset))
            self.assertEqual(len(data), expected_frames)
            self.assertGreater(np.max(np.abs(data)), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
