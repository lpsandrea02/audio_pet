"""Unit tests for audio_tools.py.

Tests both existing behaviour (baseline) and the new confused-noise
fallback behaviour (detect_melody returning [] on invalid input, and
synthesise_output generating a confused melody when given an empty log).

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
