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
    IDLE_NOISE_MIN_DELAY / IDLE_NOISE_MAX_DELAY (float): Bounds (in
        seconds) of the random spacing between consecutive idle
        character noises.
    IDLE_NOISE_MAX_COUNT (int): Fixed number of idle noises after which
        the character emits a final sad noise and then stays silent.
    SAD_NOISE_MELODY (list): Preset note sequence of the final sad
        noise (delivered with the "sad" emotion).
    INTERACTIVE_SILENCE_THRESHOLD (float): Seconds of silence after the
        latest detected note that end the user's streaming turn in
        interactive mode.
    INTERACTIVE_INTERRUPT_ANGER_THRESHOLD (int): Number of user
        interruptions after which the Audiopet responds with a forced
        angry noise instead of a melody.
    VAD_SAMPLE_RATE / VAD_FRAME_MS (int): Internal sample rate and
        frame length the WebRTC voice-activity gate runs at.
    VAD_VOICED_FRAME_RATIO / VAD_AGGRESSIVENESS (float, int): Tunables
        of the voice-activity gate in front of chunk note detection.
    LESSON_MEMORY_SIZE (int): Capacity of the separate long-term memory
        used by lesson mode: the number of finished (learned) melodies
        kept for later recall.
    LESSON_MEMORY_SING_PROBABILITY (float): Probability that a click or
        idle noise recalls a melody from the long-term lesson memory
        instead of playing a preset noise.
    LESSON_REDO_ANGRY_PROBABILITY (float): Probability that the noise
        made before a Redo Previous Section recording is the angry
        noise instead of the confused noise.
    LESSON_MISTAKE_PROBABILITY (float): Probability that the Audiopet
        makes a mistake in the middle of the melody while rehearsing it
        during Learn Melody (it then sings a confused noise and
        rehearses the whole sequence correctly a second time).
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
import webrtcvad

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
# LESSON MODE (separate long-term memory of learned melodies)
# =====================================================================

# Lesson mode teaches melodies section by section: the user records one
# section at a time, the pet sings the melody accumulated so far back,
# and once the user is satisfied the full melody is "learned" into a
# separate long-term memory (independent of the rolling short-term
# memory above) from which click and idle noises may later recall it.

# Capacity of the long-term lesson memory (the number of finished
# melodies kept; customisable) and the probability that a click or idle
# noise recalls a melody from it instead of playing a preset noise
# (controllable).
LESSON_MEMORY_SIZE = 20
LESSON_MEMORY_SING_PROBABILITY = 0.10

# Separate long-term memory of learned (finished) melodies. Kept
# strictly independent of ``short_term_memory`` so lesson recall never
# competes with the rolling short-term cache; see
# :func:`add_to_lesson_memory`.
long_term_memory = []


def combine_melody_sections(sections):
    """
    Concatenates the taught lesson sections into the full melody log.

    Lesson mode records one section at a time; before the pet sings the
    melody back (and before a finished melody is learned), the stored
    sections are flattened into a single melody log in teaching order.
    Only valid sections are merged (see :func:`_is_valid_melody_log`):
    empty, malformed, or None sections are skipped. The result is a
    fresh list of per-note copies, so later mutation of the caller's
    sections cannot corrupt the combined melody.

    Args:
        sections (list[list[dict]] or None): Taught sections in order,
            each a melody log of {"pitch": str, "duration": float}
            dicts (as produced by :func:`detect_melody` chunks merged
            with :func:`extend_melody_log`).

    Returns:
        list[dict]: The combined melody log (a new list; inputs are
        never mutated), or an empty list for empty/None input or when
        no section is valid.
    """
    if not sections:
        return []

    combined = []
    for section in sections:
        if not _is_valid_melody_log(section):
            continue
        combined.extend({**step} for step in section)
    return combined


def replace_last_section(sections, new_notes):
    """
    Replaces the latest taught section with a new (redo) recording.

    The "Redo Previous Section" flow: the latest section is swapped for
    the newly recorded notes while all earlier sections are kept in
    place — sections are replaced, never appended. Fail-safe behaviour:
    if ``new_notes`` is invalid (an empty, malformed, or failed
    recording) the previous sections are kept unchanged rather than
    wiping the taught melody, and an empty/None ``sections`` list
    simply makes the new recording the first section. The result is a
    fresh list of per-note copies; inputs are never mutated.

    Args:
        sections (list[list[dict]] or None): Taught sections in order
            (may be empty or None).
        new_notes (list[dict]): Notes detected in the redo recording.
            Invalid or empty input keeps the previous sections.

    Returns:
        list[list[dict]]: The new sections list (a new list; inputs are
        never mutated). Invalid sections are skipped.
    """
    kept = []
    for section in (sections or []):
        if _is_valid_melody_log(section):
            kept.append([{**step} for step in section])

    if not _is_valid_melody_log(new_notes):
        return kept

    new_section = [{**step} for step in new_notes]
    if kept:
        kept[-1] = new_section
    else:
        kept.append(new_section)
    return kept


def generate_lesson_reply(melody_log,
                          output_filename="system_lesson_reply.wav",
                          character="default_cat"):
    """
    Synthesises the Audiopet's lesson-mode reply and writes it to a
    dedicated lesson reply file.

    The full melody taught so far (all sections combined by
    :func:`combine_melody_sections`) is rendered in the character's
    voice with a randomly drawn happy or neutral delivery (same
    weighted ``np.random.choice`` structure as the rest of the
    behaviour logic; the draw fires before any synthesis). If the
    melody is empty or None the preset confused noise is synthesised
    with the "confused" emotion instead (the ``synthesise_output``
    fallback). The result must never be written to ``system_reply.wav``
    (reserved for the standard recording pipeline),
    ``system_noise.wav`` (reserved for pokes) or ``system_idle.wav``
    (reserved for idle noises): callers pass a separate path (e.g.
    ``data/responses/system_lesson_reply.wav``).

    Parameters:
    - melody_log (list[dict]): Full melody taught so far (as built by
      :func:`combine_melody_sections`). An empty or None log renders
      the confused noise fallback.
    - output_filename (str, optional): Path of the .wav file to write.
      Defaults to "system_lesson_reply.wav".
    - character (str, optional): Key into ``VOICE_PROFILES``; falls
      back to "default_cat" if unknown. Defaults to "default_cat".

    Returns:
        str: The emotion actually used for the synthesis ("happy",
        "none", or "confused" when the fallback fired).
    """
    emotion = str(np.random.choice(["none", "happy"], p=[0.70, 0.30]))
    return synthesise_output(melody_log,
                             output_filename=output_filename,
                             sample_rate=44100,
                             character=character,
                             emotion=emotion)


def add_to_lesson_memory(melody_log, capacity=None, memory=None):
    """
    Stores a finished (learned) melody in the long-term lesson memory.

    Reuses :func:`add_to_memory` unchanged: only valid melody logs are
    stored, the log is stored as a nested copy so later mutation of the
    caller's list cannot corrupt the memory, and once the memory
    exceeds ``capacity`` the oldest entries are evicted. Unlike the
    short-term memory, the default target is the separate module-level
    ``long_term_memory`` (capacity ``LESSON_MEMORY_SIZE``, 20) so
    learned melodies never evict or compete with the rolling
    short-term cache. ``short_term_memory`` is never touched.

    Args:
        melody_log (list[dict]): The finished melody, one dict per
            note: {"pitch": str, "duration": float}. Invalid or empty
            logs are ignored.
        capacity (int, optional): Maximum number of melodies to keep.
            Defaults to ``LESSON_MEMORY_SIZE`` (20).
        memory (list, optional): The long-term memory list to append
            to. Defaults to the module-level ``long_term_memory``.

    Returns:
        list: The updated memory list (the same object passed in).
    """
    if capacity is None:
        capacity = LESSON_MEMORY_SIZE
    if memory is None:
        memory = long_term_memory

    return add_to_memory(melody_log, capacity=capacity, memory=memory)


def pick_lesson_memory_melody(memory):
    """
    Picks a random learned melody from the long-term lesson memory.

    Thin wrapper over :func:`pick_memory_melody` with the same
    seeded-deterministic ``np.random`` draw structure as the rest of
    the behaviour logic; kept as a named entry point so lesson recall
    stays independently traceable (and re-seedable) from short-term
    recall.

    Args:
        memory (list): List of stored (learned) melody logs (as
            produced by :func:`add_to_lesson_memory`).

    Returns:
        list[dict]: One randomly chosen learned melody, or None if the
        memory is empty or None (in which case the caller must fall
        back to the default noise behaviour).
    """
    return pick_memory_melody(memory)


# Probability that the Audiopet makes a mistake in the middle of the
# melody while rehearsing it during Learn Melody (controllable): it
# then sings a confused noise and rehearses the whole sequence
# correctly a second time before the final happy noise.
LESSON_MISTAKE_PROBABILITY = 0.25

# Probability that the noise made before a Redo Previous Section
# recording is the preset angry noise instead of the preset confused
# noise (controllable).
LESSON_REDO_ANGRY_PROBABILITY = 0.4


def _note_name_to_midi(note_str):
    """
    Converts a standard scientific pitch note name to a MIDI number.

    Inverse of :func:`midi_to_note_name`: "A4" maps back to 69.

    Args:
        note_str (str): Note name in scientific pitch notation
            (e.g. "C5", "F#4").

    Returns:
        int: MIDI note number, or 0 for unparseable input.
    """
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                       'G', 'G#', 'A', 'A#', 'B']
    try:
        note_str = str(note_str).strip()
        note_name = note_str[:-1]
        octave = int(note_str[-1])
        return chromatic_scale.index(note_name) + (octave + 1) * 12
    except (ValueError, IndexError, TypeError):
        return 0


def make_mistake_melody(melody_log, semitone_shifts=(-2, -1, 1, 2)):
    """
    Returns a copy of the melody with one middle note deliberately
    mistuned, as the "mistake" the Audiopet makes while rehearsing a
    melody during Learn Melody.

    The note closest to the middle of the melody is transposed by a
    random non-zero semitone shift drawn from ``semitone_shifts`` (the
    same seeded-deterministic ``np.random`` draw structure as the rest
    of the behaviour logic), so the rehearsed rendition sounds audibly
    wrong but keeps every other note — and every duration — intact.
    Only notes with a parseable pitch can be corrupted; if the melody
    carries none, an unchanged copy is returned.

    Fail-safe behaviour: empty, malformed, or None input returns an
    empty list (the confused-noise fallback case), and the input is
    never mutated.

    Args:
        melody_log (list[dict]): Melody to corrupt, one dict per note:
            {"pitch": str, "duration": float}.
        semitone_shifts (tuple, optional): Non-zero semitone shifts the
            middle note may be transposed by. Defaults to (-2, -1, 1, 2).

    Returns:
        list[dict]: A fresh melody log (a new list of new note dicts)
        with exactly one middle note transposed, or an empty list for
        empty/invalid input.
    """
    if not _is_valid_melody_log(melody_log):
        return []

    mistake_log = [{**step} for step in melody_log]

    # Only notes with a real pitch can be mistuned (rest markers and
    # unparseable pitches are left alone).
    singable_indices = [
        index for index, step in enumerate(mistake_log)
        if _get_frequency(step.get("pitch")) > 0.0
    ]
    if not singable_indices:
        return mistake_log

    middle = len(mistake_log) / 2.0
    target_index = min(singable_indices,
                       key=lambda index: abs((index + 0.5) - middle))

    shift = int(np.random.choice(list(semitone_shifts)))
    midi_number = _note_name_to_midi(mistake_log[target_index]["pitch"])
    if midi_number > 0:
        mistake_log[target_index]["pitch"] = midi_to_note_name(
            max(1, midi_number + shift))
    return mistake_log


def should_lesson_mistake(mistake_probability=None):
    """
    Decides whether the Audiopet makes a mistake while rehearsing a
    melody during Learn Melody.

    A plain ``np.random.random()`` probability draw decides, keeping
    the behaviour deterministic under a pinned seed (same structure as
    the other probability gates).

    Args:
        mistake_probability (float, optional): Probability of the
            mistake. Defaults to ``LESSON_MISTAKE_PROBABILITY``.

    Returns:
        bool: True if the rehearsal should contain the mid-melody
        mistake (confused noise and a correct second rehearsal follow).
    """
    if mistake_probability is None:
        mistake_probability = LESSON_MISTAKE_PROBABILITY
    return bool(np.random.random() < mistake_probability)


def pick_lesson_redo_noise(angry_probability=None):
    """
    Picks which preset noise the Audiopet makes before a Redo Previous
    Section recording starts.

    With probability ``angry_probability`` the preset angry noise is
    picked, otherwise the preset confused noise (plain
    ``np.random.random()`` draw, same seeded-deterministic structure as
    the other behaviour logic).

    Args:
        angry_probability (float, optional): Probability of the angry
            noise. Defaults to ``LESSON_REDO_ANGRY_PROBABILITY``.

    Returns:
        str: "angry" or "confused".
    """
    if angry_probability is None:
        angry_probability = LESSON_REDO_ANGRY_PROBABILITY
    return "angry" if np.random.random() < angry_probability else "confused"


def generate_lesson_noise(noise_kind, character="default_cat",
                          output_filename="system_lesson_noise.wav"):
    """
    Synthesises one preset lesson-mode event noise and writes it to a
    dedicated lesson noise file.

    Lesson mode needs a small family of one-off preset noises outside
    the taught-melody pipeline: the confused or angry noise made before
    a Redo Previous Section recording, the sad noise made when the user
    forgets the melody, the confused noise after a rehearsal mistake,
    and the final happy noise at the end of the Learn Melody rehearsal.
    Each is rendered in the character's voice from its preset melody
    (``ANGRY_NOISE_MELODY``, ``CONFUSED_MELODY``, ``SAD_NOISE_MELODY``
    or ``HAPPY_NOISE_MELODY``) with the matching emotion. The result
    must never be written to ``system_reply.wav`` (standard pipeline),
    ``system_noise.wav`` (pokes) or ``system_idle.wav`` (idle noises):
    callers pass a separate path (e.g.
    ``data/responses/system_lesson_noise.wav``).

    Fail-safe behaviour: an unknown ``noise_kind`` falls back to the
    confused noise instead of raising.

    Args:
        noise_kind (str): One of "angry", "confused", "sad", "happy".
        character (str, optional): Key into ``VOICE_PROFILES``; falls
            back to "default_cat" if unknown. Defaults to "default_cat".
        output_filename (str, optional): Path of the .wav file to
            write. Defaults to "system_lesson_noise.wav".

    Returns:
        str: The emotion actually used for the synthesis (matching the
        requested noise kind, or "confused" for the fallback).
    """
    noise_presets = {
        "angry": (ANGRY_NOISE_MELODY, "angry"),
        "confused": (CONFUSED_MELODY, "confused"),
        "sad": (SAD_NOISE_MELODY, "sad"),
        "happy": (HAPPY_NOISE_MELODY, "happy"),
    }
    if noise_kind not in noise_presets:
        noise_kind = "confused"

    melody_log, emotion = noise_presets[noise_kind]
    synthesise_output(melody_log,
                      output_filename=output_filename,
                      sample_rate=44100,
                      character=character,
                      emotion=emotion)
    return emotion


# =====================================================================
# IDLE CHARACTER NOISES
# =====================================================================

# Random spacing between consecutive idle character noises: every idle
# delay is drawn uniformly from [IDLE_NOISE_MIN_DELAY,
# IDLE_NOISE_MAX_DELAY] seconds (controllable).
IDLE_NOISE_MIN_DELAY = 30.0
IDLE_NOISE_MAX_DELAY = 300.0

# Fixed number of idle noises the character may emit per idle session;
# after this many noises a final sad noise plays once and the character
# stays silent until the user interacts again (controllable).
IDLE_NOISE_MAX_COUNT = 3

# Probability that an idle noise recalls a randomly chosen melody from
# short-term memory instead of drawing one of the preset idle noises
# (controllable; skipped entirely when the memory is empty).
IDLE_MEMORY_PROBABILITY = 0.25

# Preset melody synthesised as the final sad idle noise, delivered with
# the "sad" emotion once IDLE_NOISE_MAX_COUNT idle noises have played.
SAD_NOISE_MELODY = [
    {'pitch': 'B5',  'duration': 0.25},
    {'pitch': 'A#5', 'duration': 0.25},
    {'pitch': 'A5',  'duration': 0.25},
    {'pitch': 'G#5', 'duration': 0.25},
    {'pitch': 'G5',  'duration': 0.25},
]


def get_idle_noise_delay(min_delay=None, max_delay=None):
    """
    Draws the random waiting time before the next idle character noise.

    Each idle noise is randomly spaced: the delay is drawn uniformly
    between the minimum and maximum idle-noise delays, so consecutive
    idle noises never arrive at a fixed interval. The bounds are
    controllable per call.

    Args:
        min_delay (float, optional): Lower bound of the delay in
            seconds. Defaults to ``IDLE_NOISE_MIN_DELAY``.
        max_delay (float, optional): Upper bound of the delay in
            seconds. Defaults to ``IDLE_NOISE_MAX_DELAY``.

    Returns:
        float: Delay in seconds. Fail-safe: an inverted (min > max)
        range is swapped rather than raising, and an equal-width range
        returns that exact value, so callers can never crash.
    """
    if min_delay is None:
        min_delay = IDLE_NOISE_MIN_DELAY
    if max_delay is None:
        max_delay = IDLE_NOISE_MAX_DELAY

    min_delay = float(min_delay)
    max_delay = float(max_delay)

    if min_delay > max_delay:
        min_delay, max_delay = max_delay, min_delay

    if min_delay == max_delay:
        return min_delay

    return float(np.random.uniform(min_delay, max_delay))


def pick_idle_noise(memory=None, memory_probability=None):
    """
    Picks the melody and emotion for one idle character noise.

    The noise is drawn from the idle-noise option pool: the preset
    happy melody (``HAPPY_NOISE_MELODY``), the preset confused melody
    (``CONFUSED_MELODY``), the preset angry melody
    (``ANGRY_NOISE_MELODY``), or — with ``memory_probability`` and only
    while the short-term memory is non-empty — a randomly chosen
    previously heard melody (via :func:`pick_memory_melody`) sung with
    a happy or neutral delivery (same weighted ``np.random.choice``
    structure as the click noise). Uses the same seeded-deterministic
    ``np.random`` draw structure as the rest of the behaviour logic.

    Args:
        memory (list, optional): The short-term memory (list of stored
            melody logs, as produced by :func:`add_to_memory`). An
            empty or None memory disables memory recall entirely.
            Defaults to None.
        memory_probability (float, optional): Probability that the idle
            noise is a memory recall instead of a preset noise.
            Defaults to ``IDLE_MEMORY_PROBABILITY``.

    Returns:
        tuple: ``(melody_log, emotion)`` where ``melody_log`` is one of
        the preset melodies or a memory log, and ``emotion`` is the
        delivery ("happy", "none", "confused", or "angry").
    """
    if memory_probability is None:
        memory_probability = IDLE_MEMORY_PROBABILITY

    if memory and np.random.random() < memory_probability:
        melody_log = pick_memory_melody(memory)
        emotion = str(np.random.choice(["happy", "none"], p=[0.70, 0.30]))
        return melody_log, emotion

    preset_options = [
        (HAPPY_NOISE_MELODY, "happy"),
        (CONFUSED_MELODY, "confused"),
        (ANGRY_NOISE_MELODY, "angry"),
    ]
    return preset_options[int(np.random.randint(len(preset_options)))]


def should_emit_idle_noise(idle_count):
    """
    Decides whether another idle character noise may still be emitted.

    Idle noises are emitted until ``idle_count`` reaches
    ``IDLE_NOISE_MAX_COUNT``; at that point the final sad noise plays
    instead and no further idle noises are generated (the caller must
    stop scheduling until the user interacts again).

    Args:
        idle_count (int): How many idle noises have already been
            emitted in the current idle session.

    Returns:
        bool: True if another regular idle noise is allowed.
    """
    return idle_count < IDLE_NOISE_MAX_COUNT


def generate_idle_noise(character="default_cat", idle_count=0,
                        output_filename="system_idle.wav",
                        memory=None, memory_probability=None):
    """
    Synthesises one idle character noise and writes it to a dedicated
    noise file.

    While ``should_emit_idle_noise(idle_count)`` allows it, a melody
    and emotion are drawn via :func:`pick_idle_noise` (preset
    happy/confused/angry noise or a short-term memory recall) and
    rendered in the character's voice. Once ``idle_count`` reaches
    ``IDLE_NOISE_MAX_COUNT`` the preset sad noise
    (``SAD_NOISE_MELODY``) is rendered with the "sad" emotion instead,
    flagged as final: the caller must not schedule any further idle
    noises after it. The result is written to a dedicated idle file
    (e.g. ``data/responses/system_idle.wav``) so it never touches
    ``system_reply.wav`` (reserved for direct user-to-audiopet
    interactions) or ``system_noise.wav`` (reserved for pokes).

    Parameters:
    - character (str, optional): Key into ``VOICE_PROFILES``; falls
      back to "default_cat" if unknown. Defaults to "default_cat".
    - idle_count (int, optional): How many idle noises have already
      been emitted in the current idle session. Defaults to 0.
    - output_filename (str, optional): Path of the .wav file to write.
      Defaults to "system_idle.wav".
    - memory (list, optional): The short-term memory (list of stored
      melody logs, as produced by :func:`add_to_memory`). An empty or
      None memory disables memory recall entirely. Defaults to None.
    - memory_probability (float, optional): Probability that the idle
      noise is a memory recall. Defaults to
      ``IDLE_MEMORY_PROBABILITY``.

    Returns:
        tuple: ``(emotion, is_final)`` where ``emotion`` is the noise
        emotion actually synthesised ("happy", "none", "confused",
        "angry", or "sad") for the frontend animation, and ``is_final``
        is True only when the final sad noise was rendered (no further
        idle noises may follow).
    """
    if not should_emit_idle_noise(idle_count):
        emotion = synthesise_output(SAD_NOISE_MELODY,
                                    output_filename=output_filename,
                                    sample_rate=44100,
                                    character=character,
                                    emotion="sad")
        return emotion, True

    melody_log, emotion = pick_idle_noise(memory=memory,
                                          memory_probability=memory_probability)
    synthesise_output(melody_log,
                      output_filename=output_filename,
                      sample_rate=44100,
                      character=character,
                      emotion=emotion)
    return emotion, False

# =====================================================================
# INTERACTIVE MODE (full-duplex streaming conversation)
# =====================================================================

# Silence (in seconds, controllable) measured after the latest detected
# note that ends the user's turn: once the streaming loop measures this
# much silence the Audiopet sings back the accumulated melody.
INTERACTIVE_SILENCE_THRESHOLD = 1.0

# Number of times the user may interrupt the Audiopet's singing (by
# producing new notes while it speaks) before it responds with a forced
# angry noise instead of a melody; the counter resets after it fires.
INTERACTIVE_INTERRUPT_ANGER_THRESHOLD = 3

# ---- WebRTC voice-activity gate for streamed chunks ----

# Internal sample rate the VAD runs at (WebRTC VAD only supports
# 8000/16000/32000/48000 Hz; chunks are resampled down to this rate
# before frame classification).
VAD_SAMPLE_RATE = 16000

# Length of the frames the VAD classifies (WebRTC VAD only supports
# 10/20/30 ms frames).
VAD_FRAME_MS = 30

# Fraction of 30 ms frames a chunk must have flagged as speech for the
# chunk to count as voiced (lenient default so quiet humming is not
# dropped; raise it to reject noisier environments).
VAD_VOICED_FRAME_RATIO = 0.1

# Default VAD aggressiveness (0 least aggressive ... 3 most aggressive
# noise filtering).
VAD_AGGRESSIVENESS = 3


def _load_resampled_float(input_wav, target_sample_rate=VAD_SAMPLE_RATE):
    """
    Loads an audio chunk and resamples it to the VAD sample rate.

    Decodes any container librosa supports (browser uploads may be
    mislabeled containers) down to mono float samples at
    ``target_sample_rate``.

    Args:
        input_wav (str): Path to the audio chunk file.
        target_sample_rate (int, optional): Sample rate to resample to.
            Defaults to ``VAD_SAMPLE_RATE``.

    Returns:
        numpy.ndarray or None: Mono float samples in [-1.0, 1.0] at the
        target rate, or None when the input is missing, silent-readable
        as empty, or undecodable (fail-safe for the streaming loop).
    """
    try:
        with warnings.catch_warnings():
            # Compressed containers decode through audioread's
            # deprecated fallback; keep the streaming log readable.
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            data, sr = librosa.load(input_wav, sr=None, mono=True)
    except Exception:
        return None

    data = np.asarray(data, dtype=np.float32)
    sr = int(sr)
    if sr != target_sample_rate:
        try:
            data = librosa.resample(data, orig_sr=sr,
                                    target_sr=target_sample_rate)
        except (TypeError, ValueError):
            return None
    return data


def is_chunk_voiced(input_wav, aggressiveness=None, voiced_ratio=None):
    """
    Decides whether a streamed audio chunk contains voiced activity.

    Used as the gate in front of melody detection during interactive
    mode: only chunks classified as voiced are note-analysed, while
    unvoiced chunks (background noise, silence) skip detection so they
    can cleanly accumulate into the turn-end silence clock. The chunk
    is resampled to ``VAD_SAMPLE_RATE`` and classified frame by frame
    with the WebRTC VAD; the chunk counts as voiced when the fraction
    of speech frames reaches ``voiced_ratio`` (tuned for humming and
    singing, which are voiced but not fully speech-like).

    Fail-safe behaviour: missing, empty, or undecodable input yields
    False (the streaming loop treats the chunk as silence) instead of
    crashing; individual malformed frames are skipped.

    Args:
        input_wav (str): Path to the audio chunk file.
        aggressiveness (int, optional): VAD aggressiveness 0-3
            (3 = most aggressive noise rejection). Defaults to
            ``VAD_AGGRESSIVENESS``.
        voiced_ratio (float, optional): Required fraction of voiced
            30 ms frames, in [0.0, 1.0]. Defaults to
            ``VAD_VOICED_FRAME_RATIO``.

    Returns:
        bool: True if the chunk is voiced enough to note-analyse.
    """
    if aggressiveness is None:
        aggressiveness = VAD_AGGRESSIVENESS
    if voiced_ratio is None:
        voiced_ratio = VAD_VOICED_FRAME_RATIO

    data = _load_resampled_float(input_wav)
    if data is None or len(data) == 0:
        return False

    frame_length = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)
    usable_length = len(data) - (len(data) % frame_length)
    if usable_length < frame_length:
        return False

    frames = (np.clip(data[:usable_length], -1.0, 1.0)
              * 32767).astype(np.int16).reshape(-1, frame_length)

    try:
        vad = webrtcvad.Vad(int(aggressiveness))
    except (TypeError, ValueError):
        return False

    voiced_frames = 0
    for frame in frames:
        try:
            if vad.is_speech(frame.tobytes(), VAD_SAMPLE_RATE):
                voiced_frames += 1
        except Exception:
            continue

    total_frames = len(frames)
    return (voiced_frames / float(total_frames)) >= float(voiced_ratio)


def is_user_turn_ended(silence_seconds, threshold=None):
    """
    Decides whether the user's streaming turn has ended (they stopped
    making notes for long enough).

    Interactive mode streams user audio in chunks; after each chunk the
    caller measures how much silence has passed since the latest
    detected note. When that silence reaches the threshold the turn is
    considered over and the Audiopet sings back the melody accumulated
    so far. Fail-safe: unreadable (None, non-numeric) silence values
    return False so the turn simply stays active rather than crashing
    the streaming loop.

    Args:
        silence_seconds (int or float or any): Seconds of silence since
            the latest detected note.
        threshold (int or float, optional): Silence threshold in
            seconds. Defaults to ``INTERACTIVE_SILENCE_THRESHOLD``.

    Returns:
        bool: True if the silence meets or exceeds the threshold.
    """
    if threshold is None:
        threshold = INTERACTIVE_SILENCE_THRESHOLD

    if (isinstance(silence_seconds, bool)
            or not isinstance(silence_seconds, (int, float))):
        return False

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return False

    return float(silence_seconds) >= threshold


def should_interrupt_angry(interrupt_count, threshold=None):
    """
    Decides whether the Audiopet is fed up with being interrupted.

    While the user streams audio during the Audiopet's reply, every
    newly detected note counts as an interruption. Once the count
    reaches the threshold the random melody response is abandoned and a
    forced angry noise plays instead; the caller must reset the counter
    to 0 after it fires. Fail-safe: unreadable counts return False so
    the streaming loop never crashes.

    Args:
        interrupt_count (int or float or any): How many times the user
            has interrupted the Audiopet during the current session.
        threshold (int, optional): Count at which the angry noise is
            forced. Defaults to
            ``INTERACTIVE_INTERRUPT_ANGER_THRESHOLD``.

    Returns:
        bool: True if the count reaches the anger threshold.
    """
    if threshold is None:
        threshold = INTERACTIVE_INTERRUPT_ANGER_THRESHOLD

    if (isinstance(interrupt_count, bool)
            or not isinstance(interrupt_count, (int, float))):
        return False

    try:
        threshold = int(threshold)
    except (TypeError, ValueError):
        return False

    return int(interrupt_count) >= threshold


def extend_melody_log(running_log, new_notes):
    """
    Merges chunk-detected notes into the running turn melody.

    Each streamed audio chunk may contribute newly detected notes;
    those notes are appended to the melody accumulated so far in the
    current user turn. Only valid note sequences are merged (see
    :func:`_is_valid_melody_log`): malformed or empty chunk notes leave
    the running melody unchanged, and an invalid running log is treated
    as empty. The result is a fresh list of per-note copies, so later
    mutation of either input cannot corrupt the merged melody.

    Args:
        running_log (list[dict]): Melody accumulated in the current
            turn, one dict per note:
            {"pitch": str, "duration": float}.
        new_notes (list[dict]): Notes detected in the latest audio
            chunk. Invalid, empty, or None input is ignored.

    Returns:
        list[dict]: The merged melody log (a new list; inputs are
        never mutated).
    """
    merged = ([{**step} for step in running_log]
              if _is_valid_melody_log(running_log) else [])

    if not _is_valid_melody_log(new_notes):
        return merged

    merged.extend({**step} for step in new_notes)
    return merged


def generate_interrupt_reply(melody_log,
                             output_filename="system_interactive_reply.wav",
                             character="default_cat"):
    """
    Synthesises the Audiopet's interactive-mode reply for a finished
    user turn and writes it to a dedicated reply file.

    The melody accumulated during the user's streaming turn is rendered
    in the character's voice with a randomly drawn happy or neutral
    delivery (same weighted ``np.random.choice`` structure as the rest
    of the behaviour logic). If the accumulated melody is empty the
    preset confused noise is synthesised with the "confused" emotion
    instead (the ``synthesise_output`` fallback), so an
    invalid/notes-less turn still gets an audible response. The result
    must never be written to ``system_reply.wav``: that file is
    reserved for the standard recording pipeline, so callers pass a
    separate path (e.g. ``data/responses/system_interactive_reply.wav``).

    Parameters:
    - melody_log (list[dict]): Melody accumulated during the user's
      turn (as built by :func:`extend_melody_log`). An empty or None
      log renders the confused noise fallback.
    - output_filename (str, optional): Path of the .wav file to write.
      Defaults to "system_interactive_reply.wav".
    - character (str, optional): Key into ``VOICE_PROFILES``; falls
      back to "default_cat" if unknown. Defaults to "default_cat".

    Returns:
        str: The emotion actually used for the synthesis ("happy",
        "none", or "confused" when the fallback fired).
    """
    emotion = str(np.random.choice(["none", "happy"], p=[0.70, 0.30]))
    return synthesise_output(melody_log,
                             output_filename=output_filename,
                             sample_rate=44100,
                             character=character,
                             emotion=emotion)


def generate_interrupt_angry_noise(
        character="default_cat",
        output_filename="system_interactive_angry.wav"):
    """
    Synthesises the forced angry noise for too many interruptions and
    writes it to a dedicated file.

    When the user interrupts the Audiopet's singing more than
    ``INTERACTIVE_INTERRUPT_ANGER_THRESHOLD`` times during an
    interactive session, the pet abandons the melody response and
    instead renders the preset angry noise (``ANGRY_NOISE_MELODY``)
    with the "angry" emotion. The caller must reset the interrupt
    counter after this fires. Like the other interactive outputs, the
    noise is written to its own dedicated file so it never touches
    ``system_reply.wav``.

    Parameters:
    - character (str, optional): Key into ``VOICE_PROFILES``; falls
      back to "default_cat" if unknown. Defaults to "default_cat".
    - output_filename (str, optional): Path of the .wav file to write.
      Defaults to "system_interactive_angry.wav".

    Returns:
        str: The emotion actually synthesised ("angry").
    """
    synthesise_output(ANGRY_NOISE_MELODY,
                      output_filename=output_filename,
                      sample_rate=44100,
                      character=character,
                      emotion="angry")
    return "angry"


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
