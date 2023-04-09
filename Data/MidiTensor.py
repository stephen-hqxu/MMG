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
from enum import IntEnum

class MidiTensor:
    """
    @brief MidiTensor is a tensor data representation of a MIDI file.

    The MIDI tensor is a 2D array (for now, can be changed in the future if we need more information for training)
    where the x-axis represents the time and y-axis is pitch of note.
    Time is discretised into time step, and the resolution of time step is determined from the quantisation of MIDI file.
    The number of pitch is fixed; for our application, we focus mainly on piano music,
    since there are 88 notes on a common domestic piano, we limit the y-axis to 88 dimensions.

    The tensor memory is formatted by spanning the note property (i.e., velocity) across the start and end time of a note.
    Essentially, the process converts MIDI file into piano roll.
    """
    NOTE_START: int = pretty_midi.note_name_to_number("A0")
    """
    The MIDI note number of the first note, corresponds to the lowest note on a standard domestic piano, which is `A0`.
    """
    NOTE_COUNT: int = 88
    """
    The exact number of note supported, based on the number of note on a standard domestic piano.
    """

    class TensorType(IntEnum):
        """
        @brief Specifies the type of tensor memory.
        """
        VELOCITY = 0x00

    def __init__(this, midi: PrettyMIDI):
        """
        @brief Initialise a MIDI tensor instance from a MIDI file.
        
        @param midi The MIDI data to be loaded from.
        It is assumed that the MIDI file only contains a single instrument of piano.
        It does not matter if tracks are merged into a single channel.
        """
        # extract all note information from the MIDI data into flattened arrays
        note: List[Tuple[float, float, int, int]] = [(note.start, note.end, note.velocity, note.pitch)
            for instrument in midi.instruments for note in instrument.notes]
        # sort the note by start time; this is to make sure the behaviour is deterministic when dealing with different MIDI inputs.
        note = sorted(note, key = lambda n : n[0])
        
        # deduce the number of time step; this allows removal of empty time at the start and end of the matrix
        timeStart: int = midi.time_to_tick(note[0][0]) # inclusive
        timeEnd: int = midi.time_to_tick(note[-1][1]) # exclusive
        totalTimeStep: int = timeEnd - timeStart
        # allocate tensor memory
        this.Velocity: torch.Tensor = torch.zeros((totalTimeStep, MidiTensor.NOTE_COUNT), dtype = torch.uint8)
        
        # format notes into tensor matrix
        for (start, end, velocity, pitch) in note:
            # time discretisation
            tick_start: int = midi.time_to_tick(start) - timeStart
            tick_end: int = midi.time_to_tick(end) - timeStart
            if tick_end <= tick_start:
                # invalid note, ignore
                continue

            pitch_idx: int = MidiTensor.pitchToIndex(pitch)

            this.Velocity[tick_start:tick_end, pitch_idx] = velocity

    @staticmethod
    def pitchToIndex(pitch: int) -> int:
        """
        @brief Convert the pitch of a note to the index of an array.

        @param pitch The MIDI pitch number.
        No checking is done against whether the pitch is outside the supported pitch.
        @return The index of the pitch.
        """
        return pitch - MidiTensor.NOTE_START
    
    def visualise(this, time_range: Tuple[int, int], tensor_type: TensorType, colour_name: str) -> np.ndarray:
        """
        @brief Visualise a given type of tensor memory by displaying the piano roll.

        @param range The start and end time in tick of the memory to be visualised.
        @param type Specifies the type of tensor memory to be displayed.
        @param colour_name The string colour name of the MIDI notes.
        @return An image of piano roll.
        """
        tick_start, tick_end = time_range
        colour_rgb: Tuple[float, float, float] = colors.to_rgb(colour_name)

        # select memory to be visualised
        matrix: torch.Tensor
        match(tensor_type):
            case MidiTensor.TensorType.VELOCITY:
                matrix = this.Velocity

        # copy the array to avoid sharing storage
        # TODO: may need to detach from device memory first if CUDA is used in the future
        image: np.ndarray = matrix[tick_start:tick_end].numpy().copy()
        # create RGB image, scale the intensity of colour by the value on the image
        # most MIDI data has range [0, 127], need to scale to [0, 255] to get full brightness
        image = np.repeat(image[:,:, np.newaxis], 3, axis = 2) * colour_rgb * (255.0 / 127.0)
        # round the pixel to a proper fixed-point colour format
        image = image.round().astype("uint8")
        # swap the time and pitch axis to make the image more intuitive
        image = np.swapaxes(image, 0, 1)
        return image