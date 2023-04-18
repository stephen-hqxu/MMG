from pretty_midi import PrettyMIDI

import numpy as np

import statistics
import pickle

from typing import List, Tuple

class MidiNoteRepresentation:
    """
    @brief The note representation of a MIDI file.
    All time unit are represented as integers, in tick.
    Note data are sorted by their start time.
    """
    CONTROLLER_DAMPER: int = 64
    """
    The controller number of the damper pedal.
    """

    def __init__(this, midi: PrettyMIDI):
        """
        @brief Load a MIDI file and convert it to note representation.

        @param midi The MIDI data to be loaded. 
        It is assumed that the MIDI file only contains a single instrument of piano.
        It does not matter if tracks are merged into a single channel.
        """
        note_event: List[Tuple[int, int, int, int]] = list() # start, end, velocity, pitch
        damper_event: List[Tuple[int, int]] = list() # time, value
        # extract all note information from the MIDI data into flattened arrays
        for instrument in midi.instruments:
            note_event.extend([(midi.time_to_tick(n.start), midi.time_to_tick(n.end), n.velocity, n.pitch) for n in instrument.notes])
            damper_event.extend([(midi.time_to_tick(cc.time), cc.value) for cc in instrument.control_changes \
                if cc.number == MidiNoteRepresentation.CONTROLLER_DAMPER])
        
        this.Resolution: int = midi.resolution
        """
        MIDI resolution in PPQ.
        """
        # sort the note by start time; this is to make sure the behaviour is deterministic when dealing with different MIDI inputs.
        # if start time is the same (for instance a chord), then sort by pitch.
        this.NoteEvent: np.ndarray = np.array(sorted(note_event, key = lambda n : (n[0], n[3])), dtype = np.uint32)
        """
        A matrix of N*4, where N is the total number of note.
        Each row contains 4 elements: start time, end time, velocity and pitch.
        """
        this.DamperEvent: np.ndarray = np.array(sorted(damper_event, key = lambda d : d[0]), dtype = np.uint32)
        """
        A matrix of N*2, where N is the same as the number of note.
        Each row contains 2 elements: time and value.
        """

        # deduce the valid start time of the MIDI; this allows removal of empty time at the beginning
        timeStart: int = min(
            this.NoteEvent[0][0],
            this.DamperEvent[0][0]
        )
        this.offsetTime(-timeStart)

    @classmethod
    def fromMidiCollection(cls, midi_collection: List[PrettyMIDI], padding: int = 0):
        """
        @brief Convert a collection of MIDI to note representation, and concatenate them in order.

        @param midi_collection An array of MIDI inputs.
        @param padding Specify time in tick of empty time to be padded before concatenating the next MIDI.
        """
        # convert all MIDI to note representation
        note_collection: List[MidiNoteRepresentation] = [cls(midi) for midi in midi_collection]
        
        # move all new notes so they are strictly behind the current MIDI, hence no need to sort everything again
        cumulative_playback: int = 0
        for n in note_collection:
            current_playback: int = n.playbackTime() + padding
            n.offsetTime(cumulative_playback)
            cumulative_playback += current_playback
        
        joint_note: MidiNoteRepresentation = note_collection[0] # to save time, just make a reference rather than a copy
        joint_note.NoteEvent = np.concatenate(tuple((n.NoteEvent for n in note_collection)), axis = 0)
        joint_note.DamperEvent = np.concatenate(tuple((n.DamperEvent for n in note_collection)), axis = 0)

        # set the resolution of the concatenated data
        joint_note.Resolution = statistics.median([n.Resolution for n in note_collection])
        return joint_note
    
    @classmethod
    def fromBinary(cls, src: str):
        """
        @brief Deserialise a binary from file to a note representation.

        @param src The source filename.
        @return The note representation.
        """
        with open(src, "rb") as note_file:
            return pickle.load(note_file)
    
    def toBinary(this, dest: str) -> None:
        """
        @brief Serialise the current note representation instance to a binary and store to local filesystem.

        @param dest The destination filename.
        """
        with open(dest, "wb") as note_file:
            pickle.dump(this, note_file, protocol = pickle.HIGHEST_PROTOCOL)
    
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