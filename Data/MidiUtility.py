"""
MIDI file utility.
"""

from pretty_midi import PrettyMIDI

import numpy as np

import pickle

from typing import List, Tuple
from dataclasses import dataclass

CONTROLLER_DAMPER: int = 64
"""
The controller number of the damper pedal.
"""

@dataclass
class NoteRepresentation:
    """
    @brief The note representation of a MIDI file.
    All time unit are represented as integers, in tick.
    """
    Resolution: int
    """
    MIDI resolution in PPQ.
    """

    NoteEvent: np.ndarray
    """
    A matrix of N*4, where N is the total number of note.
    Each row contains 4 elements: start time, end time, velocity and pitch.
    """
    DamperEvent: np.ndarray
    """
    A matrix of N*2, where N is the same as the number of note.
    Each row contains 2 elements: time and value.
    """
    
    def playbackTime(this) -> int:
        """
        @brief Get the playback time of this MIDI.
        This is the time when the last MIDI event finishes; or equivalently , one tick past the last tick in the MIDI.

        @return The playback time in tick.
        """
        return max(
            np.max(this.NoteEvent[:, 1]),
            this.DamperEvent[-1][0]
        )
    
    def offsetTime(this, offset: int) -> None:
        """
        @brief Increment the time information for all data in the note representation.

        @param offset The number of tick to move.
        """
        this.NoteEvent[:, 0:2] += offset
        this.DamperEvent[:, 0] += offset

def midiToNote(midi: PrettyMIDI) -> NoteRepresentation:
    """
    @brief Load a MIDI file and convert it to note representation.

    @param midi The MIDI data to be loaded. 
    It is assumed that the MIDI file only contains a single instrument of piano.
    It does not matter if tracks are merged into a single channel.
    @return The MIDI note representation.
    """
    note_event: List[Tuple[int, int, int, int]] = list() # start, end, velocity, pitch
    damper_event: List[Tuple[int, int]] = list() # time, value
    # extract all note information from the MIDI data into flattened arrays
    for instrument in midi.instruments:
        note_event.extend([(midi.time_to_tick(n.start), midi.time_to_tick(n.end), n.velocity, n.pitch) for n in instrument.notes])
        damper_event.extend([(midi.time_to_tick(cc.time), cc.value) for cc in instrument.control_changes if cc.number == CONTROLLER_DAMPER])
    
    # sort the note by start time; this is to make sure the behaviour is deterministic when dealing with different MIDI inputs.
    # if start time is the same (for instance a chord), then sort by pitch.
    note_event = sorted(note_event, key = lambda n : (n[0], n[3]))
    damper_event = sorted(damper_event, key = lambda d : d[0])

    # do some post processing
    repr: NoteRepresentation = NoteRepresentation(midi.resolution,
        np.array(note_event, dtype = np.uint32), np.array(damper_event, dtype = np.uint32))
    # deduce the valid start time of the MIDI; this allows removal of empty time at the beginning
    timeStart: int = min(
        repr.NoteEvent[0][0],
        repr.DamperEvent[0][0]
    )
    repr.offsetTime(-timeStart)
    return repr

def noteToBinary(note_repr: NoteRepresentation, dest: str) -> None:
    """
    @brief Serialise a note representation to binary and store to local filesystem.

    @param note_repr The note representation to be serialised.
    @param dest The destination filename.
    """
    with open(dest, "wb") as note_file:
        pickle.dump(note_repr, note_file, protocol = pickle.HIGHEST_PROTOCOL)

def binaryToNote(src: str) -> NoteRepresentation:
    """
    @brief Deserialise a binary from file to a note representation.

    @param src The source filename.
    @return The note representation.
    """
    with open(src, "rb") as note_file:
        return pickle.load(note_file)