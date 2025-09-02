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

def my_synth(n: note.Note, operator_toggle:list[bool], adsr: list = [0.1, 0.01, 0.8, 0.2]) -> np.ndarray:
    """A simple synthesizer that generates a sound wave for a given note."""
    print(n, n.pitch.frequency, n.duration.quarterLength, operator_toggle, adsr)
    
    out = sine_tone(frequency=n.pitch.frequency, duration=float(n.duration.quarterLength), amplitude=1.0)
    if operator_toggle[0]:
        out = am_synth(out, random.choice(range(2,10)))
    if operator_toggle[1]:
        out = am_synth(out, n.pitch.frequency)
    if operator_toggle[2]:
        out = fm_synth(out, n.pitch.frequency / 2 - 3, 5.0)
    if operator_toggle[3]:
        out = fm_synth(out, random.choice(range(2,10)))
    if operator_toggle[4]:
        out = am_synth(out, n.pitch.frequency / 8)
    if operator_toggle[5]:
        out = fm_synth(out, n.pitch.frequency / 2.0, 5.0)
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

def axis_progression(_durations: list[duration.Duration], operator_toggle: list[list[bool]]):
    # Not sure if i am getting this right... oh well. Exploring for funzies.
    # Axis Progression aka Axis of Awesome "Four Chord Wonder progression"
    # I V vi IV
    # For help: Circle of Fifths https://www.youtube.com/watch?v=O43EBVnwNvo
    root_chord_scale: scale.Scale = scale.MajorScale('C')
    tonic = note.Note(root_chord_scale.tonic)
    fifth = tonic.transpose(interval.Interval('P5')) # .transposePitch(root_chord.chord)    
    sixth = tonic.transpose(interval.Interval('m6')) #.transposePitch(root_chord.chord)
    fourth = tonic.transpose(interval.Interval('P4')) #.transposePitch(root_chord.chord)
    print(tonic, fifth, sixth, fourth)

    my_scaleB = scale.MajorScale(fifth)
    my_scaleC = scale.MinorScale(sixth)
    my_scaleD = scale.MajorScale(fourth)
    print(root_chord_scale, my_scaleB, my_scaleC, my_scaleD)

    # For each phrase choose 4 notes to play out of the respective chord
    # The accumulative melody should sound like the chord progression?
    phraseA = [random.choice(root_chord_scale.chord) for _ in range(4)]
    
    phraseB = [random.choice(my_scaleB.chord) for _ in range(4)]

    phraseC = [random.choice(my_scaleC.chord) for _ in range(4)]

    phraseD = [random.choice(my_scaleD.chord) for _ in range(4)]

    # Combine each phrase
    notes = phraseA + phraseB + phraseC + phraseD
    for i, n in enumerate(notes):
        n.duration = _durations[i]

    envelopes = [
        [0.1, 0.01, 0.8, 0.2],
        [0.05, 0.01, 0.8, 0.1],
        [0.1, 0.02, 0.7, 0.3],
        [0.1, 0.01, 0.8, 0.2]
    ]

    sound_parts = []
    for i, n in enumerate(notes):
        # Run the note fundamental frequency through my synth operator.
        operator_toggle_index = i % len(operator_toggle)
        synth_note = my_synth(n, operator_toggle=operator_toggle[operator_toggle_index], adsr=random.choice(envelopes))
        sound_parts.append(synth_note)

    # Concatenate into one continuous sound
    final_sound = reduce(lambda a, b: np.concatenate((a, b)), sound_parts)
    
    return final_sound

def main():
    # The relative ratio of options influences the relative ratio of outcome.
    duration_choices = [2.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    operator_toggle_choices = [True, True,True,True,True,True,True, False, False]
    
    # Setup the same set of 16 durations that gets the rhythm applied to each progression.
    _durations = [duration.Duration(1.0 / random.choice(duration_choices)) for _ in range(16)]

    # Toggle synth operators every 4 notes
    _operator_toggles = [[random.choice(operator_toggle_choices) for _ in range(6)] for _ in range(3)]

    final_sound = reduce(lambda a, b: np.concatenate((a, b)), [axis_progression(_durations, _operator_toggles) for _ in range(32)])
    sd.play(final_sound, samplerate=44100)
    sd.wait()

if __name__ == "__main__":
    main()
