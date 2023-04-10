"""
Conversion and format between MIDI data and PyTorch tensor.
"""

# PyTorch
import torch
# MIDI
import pretty_midi
from pretty_midi import PrettyMIDI

import numpy as np
from matplotlib import colors

# System
from typing import List, Tuple

class MidiTensor:
    """
    @brief MidiTensor is a tensor data representation of a MIDI file.

    The MIDI tensor is a 2D array (for now, can be changed in the future if we need more information for training)
    where the x-axis represents the time and y-axis is pitch of note. The value of each matrix entry represents velocity.
    Time is discretised into time step, and the resolution of time step is determined from the quantisation of MIDI file.
    The number of pitch is fixed; for our application, we focus mainly on piano music,
    since there are 88 notes on a common domestic piano, we limit the y-axis to 88 dimensions.

    The tensor memory is formatted by spanning the note property (i.e., velocity) across the start and end time of a note.
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

    CONTROLLER_DAMPER: int = 64
    """
    The controller number of the damper pedal.
    """

    def __init__(this, midi: PrettyMIDI):
        """
        @brief Initialise a MIDI tensor instance from a MIDI file.
        
        @param midi The MIDI data to be loaded from.
        It is assumed that the MIDI file only contains a single instrument of piano.
        It does not matter if tracks are merged into a single channel.
        """
        note_event: List[Tuple[int, int, int, int]] = list() # start, end, velocity, pitch
        damper_event: List[Tuple[int, int]] = list() # time, value
        # extract all note information from the MIDI data into flattened arrays
        for instrument in midi.instruments:
            note_event.extend([(midi.time_to_tick(note.start), midi.time_to_tick(note.end), note.velocity, note.pitch) for note in instrument.notes])
            damper_event.extend([(midi.time_to_tick(cc.time), cc.value) for cc in instrument.control_changes if cc.number == MidiTensor.CONTROLLER_DAMPER])

        # sort the note by start time; this is to make sure the behaviour is deterministic when dealing with different MIDI inputs.
        # if start time is the same (for instance a chord), then sort by pitch.
        note_event = sorted(note_event, key = lambda n : (n[0], n[3]))
        damper_event = sorted(damper_event, key = lambda d : d[0])
        
        # deduce the number of time step; this allows removal of empty time at the start and end of the matrix
        # we want all events to line up with each other, so make sure they have the same length
        timeStart: int = min(
            note_event[0][0],
            damper_event[0][0]
        ) # inclusive
        # end time is the last note finished
        timeEnd: int = max(
            max(note_event, key = lambda n : n[1])[1],
            damper_event[-1][0]
        ) # exclusive
        totalTimeStep: int = timeEnd - timeStart
        
        this.Resolution: int = midi.resolution
        """
        The resolution of the MIDI file.
        """
        this.PianoRoll: torch.Tensor = torch.zeros((totalTimeStep, MidiTensor.NOTE_COUNT + 1), dtype = torch.uint8)
        """
        This is a piano roll representation of the data, with all necessary data encoded.

        For the second dimension, the first *note_count* arrays are for velocity, and the last array is for damper pedal.
        All derived data structures are meant to be references of this piano roll.
        """
        
        for (start, end, velocity, pitch) in note_event:
            tick_start: int = start - timeStart
            tick_end: int = end - timeStart
            if tick_end <= tick_start:
                # invalid note, ignore
                continue

            pitch_idx: int = MidiTensor.pitchToIndex(pitch)
            this.velocity()[tick_start:tick_end, pitch_idx] = velocity

        # convert control change to the actual value
        # which means the value is remained since last time set until it is changed next time
        prev_time: int = 0
        prev_value: int = 0
        for time, value in damper_event:
            this.damper()[prev_time:time] = prev_value
            prev_time = time
            prev_value = value
        # set the controller for the rest of the time
        this.damper()[prev_time:] = prev_value

    @staticmethod
    def pitchToIndex(pitch: int) -> int:
        """
        @brief Convert the pitch of a note to the index of an array.

        @param pitch The MIDI pitch number.
        No checking is done against whether the pitch is outside the supported pitch.
        @return The index of the pitch.
        """
        return pitch - MidiTensor.NOTE_START
    
    def velocity(this) -> torch.Tensor:
        """
        @brief Get the reference of velocity in the piano roll.

        @return A matrix of velocity (a.k.a. dynamic) of each MIDI note event.
        """
        return this.PianoRoll[:, 0:MidiTensor.NOTE_COUNT]
    
    def damper(this) -> torch.Tensor:
        """
        @brief Get the reference of damper pedal in the piano roll.

        @return A vector of damper pedal value at every time step.
        """
        return this.PianoRoll[:, -1]
    
    def visualiseVelocity(this, time_range: Tuple[int, int], colour_name: str) -> np.ndarray:
        """
        @brief Visualise MIDI notes by displaying their velocities.
        The intensity of the colour represents the strength of velocity.

        @param time_range The start and end time in tick of the memory to be visualised.
        @param colour_name The string colour name of the MIDI notes.
        @return An image of note velocity visualisation.
        """
        colour_rgb: Tuple[float, float, float] = colors.to_rgb(colour_name)

        # copy the array to avoid sharing storage
        # TODO: may need to detach from device memory first if CUDA is used in the future
        image: np.ndarray = this.velocity()[slice(*time_range)].numpy().copy()
        # create RGB image, scale the intensity of colour by the value on the image
        # most MIDI data has range [0, 127], need to scale to [0, 255] to get full brightness
        image = np.repeat(image[:,:, np.newaxis], 3, axis = 2) * colour_rgb * (255.0 / 127.0)
        # round the pixel to a proper fixed-point colour format
        image = image.round().astype("uint8")
        # swap the time and pitch axis to make the image more intuitive
        image = np.swapaxes(image, 0, 1)
        return image