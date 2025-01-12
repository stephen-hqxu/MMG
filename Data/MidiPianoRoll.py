from Data.MidiNoteRepresentation import MidiNoteRepresentation

import pretty_midi
from pretty_midi import PrettyMIDI

import torch
import numpy as np
from matplotlib import colors

from typing import Tuple

class MidiPianoRoll:
    """
    @brief MidiPianoRoll is a data representation of a MIDI file.

    The MIDI piano roll is a 2D array (for now, can be changed in the future if we need more information for training)
    where the x-axis represents the time and y-axis is pitch of note. The value of each matrix entry represents velocity.
    Time is discretised into time step, and the resolution of time step is determined from the quantisation of MIDI file.
    The number of pitch is fixed; for our application, we focus mainly on piano music,
    since there are 88 notes on a common domestic piano, we limit the y-axis to 88 dimensions.

    The memory is formatted by spanning the note property (i.e., velocity) across the start and end time of a note.
    Essentially, the process converts MIDI file into piano roll.

    In addition, other properties from a MIDI music are also included, such as control changes (e.g., pedal data),
    which are encoded as a 1D array, where the length is the number of time step as in the velocity matrix.
    """
    NOTE_START: int = pretty_midi.note_name_to_number("A0")
    """
    The MIDI note number of the first note, corresponds to the lowest note on a standard domestic piano, which is `A0`.
    """
    NOTE_COUNT: int = 88
    """
    The exact number of note supported, based on the number of note on a standard domestic piano.
    """

    CONTROLLER_COUNT: int = 1
    """
    The number of controller channel to be encoded into the piano roll.
    """

    NOTE_MAX_LEVEL: int = 128
    """
    The maximum of level a note can have. This defines the upper limit of each element in the piano roll.
    The minimum level of a note is always zero.
    In another word, the range of data is [0, NOTE_MAX_LEVEL).
    """
    DIMENSION_PER_TIME_STEP: int = NOTE_COUNT + CONTROLLER_COUNT
    """
    The number of note (velocity and controller data) in a single time step. This is equivalent to the dimension of `y`.
    """

    def __init__(this, midi_note: MidiNoteRepresentation):
        """
        @brief Initialise a MIDI piano roll instance.
        
        @param midi_note The MIDI data in note representation.
        """
        this.Resolution: int = midi_note.Resolution
        """
        The resolution of the MIDI file.
        """
        this.PianoRoll: np.ndarray = np.zeros((midi_note.playbackTime(), MidiPianoRoll.DIMENSION_PER_TIME_STEP), dtype = np.uint8)
        """
        This is a piano roll representation of the data, with all necessary data encoded.

        For the second dimension, the first *note_count* arrays are for velocity, and the last array is for damper pedal.
        All derived data structures are meant to be references of this piano roll.
        """
        velocityMatrix: np.ndarray = this.velocity()
        pitchStart: int = MidiPianoRoll.NOTE_START
        pitchEnd: int = pitchStart + MidiPianoRoll.NOTE_COUNT # one past the end
        for start, end, velocity, pitch in midi_note.NoteEvent:
            # all invalid notes should have been removed by pretty MIDI
            assert(start < end)
            if pitch < pitchStart or pitch >= pitchEnd:
                # skip out-of-domain pitches
                continue

            pitch_idx: int = MidiPianoRoll.pitchToIndex(pitch)
            velocityMatrix[start:end, pitch_idx] = velocity

        damperArray: np.ndarray = this.damper()
        # convert control change to the actual value
        # which means the value is remained since last time set until it is changed next time
        prev_time: int = 0
        prev_value: int = 0
        for time, value in midi_note.DamperEvent:
            damperArray[prev_time:time] = prev_value
            prev_time = time
            prev_value = value
        # set the controller for the rest of the time
        damperArray[prev_time:] = prev_value

    @classmethod
    def fromMidi(cls, midi: PrettyMIDI):
        """
        @brief Initialise a MIDI piano roll instance from a MIDI file.

        @param midi The MIDI file to be loaded from, which will be first converted MIDI note representation.
        @see MidiUtility.midiToNote()
        """
        return cls(MidiNoteRepresentation(midi))

    @staticmethod
    def pitchToIndex(pitch: int) -> int:
        """
        @brief Convert the pitch of a note to the index of an array.

        @param pitch The MIDI pitch number.
        No checking is done against whether the pitch is outside the supported pitch.
        @return The index of the pitch.
        """
        return pitch - MidiPianoRoll.NOTE_START
    
    @staticmethod
    def sliceVelocity() -> slice:
        """
        The velocity submatrix.
        """
        return slice(0, MidiPianoRoll.NOTE_COUNT)
    
    @staticmethod
    def sliceControl() -> slice:
        """
        The control submatrix.
        """
        return slice(MidiPianoRoll.NOTE_COUNT, MidiPianoRoll.DIMENSION_PER_TIME_STEP)
    
    @staticmethod
    def sliceDamper() -> slice:
        """
        The damper control within the control submatrix.
        """
        return slice(MidiPianoRoll.NOTE_COUNT, MidiPianoRoll.NOTE_COUNT + 1)
    
    def velocity(this) -> np.ndarray:
        """
        @brief Get the reference of velocity value from the internal encoded MIDI piano roll.

        @return A matrix of velocity (a.k.a. dynamic) of each MIDI note event.
        """
        return this.PianoRoll[:, MidiPianoRoll.sliceVelocity()]
    
    def damper(this) -> np.ndarray:
        """
        @brief Get the reference of damper pedal value from the internal encoded MIDI piano roll.

        @return A vector of damper pedal value at every time step.
        """
        return this.PianoRoll[:, MidiPianoRoll.sliceDamper()]
    
    def visualiseVelocity(this, time_range: Tuple[int, int], colour_name: str) -> np.ndarray:
        """
        @brief Visualise MIDI notes by displaying their velocities.
        The intensity of the colour represents the strength of velocity.

        @param time_range The start and end time in tick of the memory to be visualised.
        @param colour_name The string colour name of the MIDI notes.
        @return An image of note velocity visualisation.
        """
        colour_rgb: Tuple[float, float, float] = colors.to_rgb(colour_name)

        image: np.ndarray = this.velocity()[slice(*time_range)].copy()
        # create RGB image, scale the intensity of colour by the value on the image
        # most MIDI data has range [0, 127], need to scale to [0, 255] to get full brightness
        image = np.repeat(image[:, :, None], 3, axis = 2) * colour_rgb * (255.0 / 127.0)
        # round the pixel to a proper fixed-point colour format
        image = image.round().astype("uint8")
        # swap the time and pitch axis to make the image more intuitive
        image = np.swapaxes(image, 0, 1)
        return image
    
    def visualiseDamper(this, time_range: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief Visualise the damper control values.

        @param time_range The start and end time in tick of damper value to be visualised.
        @return The `x` and `y` axis values of the plot, where `x` is the tick scale label,
        and `y` is the damper value as each tick.
        """
        start, end = time_range
        return (np.arange(start, end), this.damper()[slice(*time_range)].copy())
    
    def viewTensor(this) -> torch.Tensor:
        """
        @brief Create a view of the piano roll as a PyTorch tensor.
        It will have the exact same data as stored in the class instance.

        @return The tensor data.
        """
        return torch.from_numpy(this.PianoRoll)