#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "sounddevice",
#   "music21"
# ]
# ///
import random
import numpy as np
import sounddevice as sd
from functools import reduce
from music21 import pitch, chord, note, stream, interval, duration, scale

# TODO: https://www.youtube.com/watch?v=eQJuaY8a-ts Explore generative music with a Lindemayer System (L-System)

def sine_tone(
        frequency: int=440,
        duration: float=1.0,
        amplitude: float=0.5,
        sample_rate: int=44100
) -> np.ndarray:
    n_samples = int(sample_rate * duration)
    time_points = np.linspace(0, duration, n_samples, endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * time_points)

def am_synth(
        modulator_wave: np.ndarray,
        carrier_freq: int=440,
        modulation_index: float=0.5,
        amplitude: float=0.5,
        sample_rate: int=44100
) -> np.ndarray:
    total_samples = modulator_wave.shape[0]
    time_points = np.arange(total_samples) / sample_rate
    carrier_wave = amplitude * np.sin(2 * np.pi * carrier_freq * time_points)
    am_wave = (1 + modulation_index * modulator_wave) * carrier_wave
    max_amplitude = np.max(np.abs(am_wave))
    am_wave /= max_amplitude
    am_wave *= amplitude
    return am_wave

def fm_synth(
        modulator_wave: np.ndarray,
        carrier_freq: int=440,
        modulation_index: float=3.0,
        amplitude: float=0.5,
        sample_rate: int=44100
) -> np.ndarray:
    total_samples = modulator_wave.shape[0]
    time_points = np.arange(total_samples) / sample_rate
    fm_wave = amplitude * np.sin(2 * np.pi * carrier_freq * time_points + modulation_index * modulator_wave)
    max_amplitude = np.max(np.abs(fm_wave))
    fm_wave /= max_amplitude
    fm_wave *= amplitude
    return fm_wave

def envelope(sound: np.ndarray, adsr: list, sample_rate: int=44100) -> np.ndarray:
    """Apply an ADSR envelope to a sound signal.

    sound: The input sound signal to which the envelope will be applied.
    adsr: A list containing the ADSR parameters [attack, decay, sustain_level, release].
    sample_rate: The sample rate of the sound signal.
    """
    # Make a copy of the original sound to avoid modifying it directly.
    _sound = sound.copy()

    # Extract Configuration
    duration = len(_sound) / sample_rate
    
    # These are meant to be durations * sample_rate to get envelope section samples.
    min_envelope_size = adsr[0] + adsr[1] + adsr[3] 

    # If the envelope spec is longer than the duration of the sound
    # Then there is no sustain section and we normalise the parameters to the length of the sound duration.
    if duration < min_envelope_size:
        # Envelope is longer than duration. 
        # Need to normalise the parameters to the duration length
        a = adsr[0] / min_envelope_size * duration
        d = adsr[1] / min_envelope_size * duration
        r = adsr[3] / min_envelope_size * duration
    else:
        a = adsr[0]
        d = adsr[1]
        r = adsr[3]

    attack_samples = int(a * sample_rate)
    decay_samples = int(d * sample_rate)
    release_samples = int(r * sample_rate)

    sustain_level = adsr[2]
    sustain_samples = len(_sound) - (attack_samples + decay_samples + release_samples)

    # Apply Envelope to signal
    # Attack
    _sound[:attack_samples] *= np.linspace(0, 1, attack_samples)
    # Decay
    _sound[attack_samples:attack_samples + decay_samples] *= np.linspace(1, sustain_level, decay_samples)
    # Sustain
    _sound[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] *= sustain_level
    # Release
    _sound[-release_samples:] *= np.linspace(sustain_level, 0, release_samples)

    return _sound

def my_synth(n: note.Note | list[note.Note] | chord.Chord, operator_toggle:list[bool], adsr: list = [0.1, 0.01, 0.8, 0.2]) -> np.ndarray:
    """A simple synthesizer that generates a sound wave for a given note."""
    
    root_note_frequency = 0
    
    # TODO: Add option of generating non-sinusoidal waveforms and maybe even harmonics of the root notes?
    if isinstance(n, chord.Chord):
        # for _note in n.notes:
        #     print(_note, _note.pitch.frequency, float(n.duration.quarterLength), operator_toggle, adsr)
        out = sum(sine_tone(frequency=_note.pitch.frequency, duration=float(n.duration.quarterLength), amplitude=1.0) for _note in n.notes)
        root_note_frequency = n.notes[0].pitch.frequency
    elif isinstance(n, list):
        # for _note in n:
        #     print(_note, _note.pitch.frequency, float(n[0].duration.quarterLength), operator_toggle, adsr)
        out = sum(sine_tone(frequency=_note.pitch.frequency, duration=float(n[0].duration.quarterLength), amplitude=1.0) for _note in n)
        root_note_frequency = n[0].pitch.frequency
    else:
        # print(n, n.pitch.frequency, n.duration.quarterLength, operator_toggle, adsr)
        out = sine_tone(frequency=n.pitch.frequency, duration=float(n.duration.quarterLength), amplitude=1.0)
        root_note_frequency = n.pitch.frequency

    out /= np.max(np.abs(out))  # Normalize to prevent clipping

    if operator_toggle[0]:
        out = am_synth(out, random.choice(range(2,10)))
    if operator_toggle[1]:
        out = am_synth(out, carrier_freq=root_note_frequency)
    if operator_toggle[2]:
        out = fm_synth(out, carrier_freq=root_note_frequency / 2 - 3, modulation_index=5.0)
    if operator_toggle[3]:
        out = fm_synth(out, carrier_freq=random.choice(range(2,10)), modulation_index=random.choice(range(2,10)))
    if operator_toggle[4]:
        out = am_synth(out, carrier_freq=root_note_frequency / 8)
    if operator_toggle[5]:
        out = fm_synth(out, carrier_freq=root_note_frequency / 2.0, modulation_index=5.0)

    out = envelope(out, adsr)
    
    return out

def chord_from_fundamental(fundamental: pitch.Pitch, harmonic_numbers: list, nearest_semitone: bool = False) -> chord.Chord:
    """Generate a chord from the given fundamental pitch."""
    harmonic_chord = chord.Chord()
    for i in harmonic_numbers:
        p = fundamental.getHarmonic(i)
        if nearest_semitone:
            p = p.midi
        harmonic_chord.add(p)
    return harmonic_chord

def axis_progression(_durations: list[duration.Duration], operator_toggle: list[list[bool]], octave: int = 5, key_of: str = 'C') -> np.ndarray:
    # Not sure if i am getting this right... oh well. Exploring for funzies.
    # Axis Progression aka Axis of Awesome "Four Chord Wonder progression"
    # I V vi IV
    # For help: Circle of Fifths https://www.youtube.com/watch?v=O43EBVnwNvo
    # I (C major): C – E – G
    # V (G major): G – B – D
    # vi (A minor): A – C – E
    # IV (F major): F – A – C

    # TODO: Allow for different octaves

    root_chord_scale: scale.Scale = scale.MajorScale(f'{key_of}{octave}')
    tonic = note.Note(root_chord_scale.tonic)
    fifth = tonic.transpose(interval.Interval('P5')) 
    sixth = tonic.transpose(interval.Interval('m6')) 
    fourth = tonic.transpose(interval.Interval('P4'))
    # print(tonic, fifth, sixth, fourth)

    my_scaleB = scale.MajorScale(fifth)
    my_scaleC = scale.MinorScale(sixth)
    my_scaleD = scale.MajorScale(fourth)
    # print(root_chord_scale, my_scaleB, my_scaleC, my_scaleD)
    # print(root_chord_scale.chord, my_scaleB.chord, my_scaleC.chord, my_scaleD.chord)
    

    phraseA = [chord.Chord([root_chord_scale.chord.notes[i] for i in [0, 2, 4]]) for _ in range(1)]

    phraseB = [chord.Chord([my_scaleB.chord.notes[i] for i in [0, 2, 4]]) for _ in range(1)]

    phraseC = [chord.Chord([my_scaleC.chord.notes[i] for i in [0, 2, 4]]) for _ in range(1)]

    phraseD = [chord.Chord([my_scaleD.chord.notes[i] for i in [0, 2, 4]]) for _ in range(1)]

    # Combine each phrase (a few times)
    repeats = (len(operator_toggle) // 4)
    print(f"{repeats=}")
    notes = (phraseA + phraseB + phraseC + phraseD) * repeats
    print(notes)
    envelopes = [
        # [0.1, 0.01, 0.8, 0.2],
        [0.05, 0.01, 0.8, 0.1],
        # [0.1, 0.02, 0.7, 0.3],
        # [0.1, 0.01, 0.8, 0.2]
    ]

    sound_parts = []
    for i, n in enumerate(notes):
        # Run the note fundamental frequency through my synth operator.
        operator_toggle_index = i % len(operator_toggle)
        duration_index = i % len(_durations)
        n.duration = _durations[duration_index]
        synth_note = my_synth(n, operator_toggle=operator_toggle[operator_toggle_index], adsr=random.choice(envelopes))
        sound_parts.append(synth_note)

    # Concatenate into one continuous sound
    final_sound = reduce(lambda a, b: np.concatenate((a, b)), sound_parts)
    
    return final_sound

def main():
    # The relative ratio of options influences the relative ratio of outcome.
    duration_choices = [2.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    operator_toggle_choices = [True, True, True, True, True, False, False, False]
    
    # Setup the same set of 16 durations that gets the rhythm applied to each progression.
    _durations = [duration.Duration(1.0 / random.choice(duration_choices)) for _ in range(16)]

    _operator_toggles = [
        # [True, False, True, True, True, True],
        # [random.choice(operator_toggle_choices) for _ in range(6)],

        # Plain Sine Wave of Chords
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],
        [False, False, False, False, False, False],

        # AM_Synth only operators for warbly / tremolo like sound
        [True, True, False, False, True, False],
        [True, True, False, False, True, False],
        [True, True, False, False, True, False],
        [True, True, False, False, True, False],

        # AM and FM Half Harmonics
        # [True, True, False, True, False, True],
        # [True, True, False, True, False, True],
        # [True, True, False, True, False, True],
        # [True, True, False, True, False, True],

        # All Operators on
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],

        # Drop the half freq -3 dissonance fm_synth
        [True, True, False, True, True, True],
        [True, True, False, True, True, True],
        [True, True, False, True, True, True],
        [True, True, False, True, True, True],

        # All Operators on
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
    ]
    key_of_list = ['C', 'G', 'D', 'A'] #['C', 'G', 'D', 'A', 'E', 'B']
    N_progressions = 12

    final_sound = reduce(
        lambda a, b: np.concatenate((a, b)), 
        [
            axis_progression(
                _durations, 
                _operator_toggles, 
                octave=(5 - (11*i+1)%3),
                key_of=key_of_list[i % len(key_of_list)]
            ) 
            for i in range(N_progressions)
        ]
    )
    quarterLengthPattern = [float(d.quarterLength) for d in _durations]
    key_and_octave_sequence = [f"{key_of_list[i % len(key_of_list)]}{(5 - (11*i+1)%3)}" for i in range(N_progressions)]
    print(f"{key_and_octave_sequence=}")
    print(f"{_operator_toggles=}")
    print(f"{quarterLengthPattern=}")

    sd.play(final_sound, samplerate=44100)
    sd.wait()

if __name__ == "__main__":
    main()
