from __future__ import print_function

import numpy as np
import sys
import argparse
import pretty_midi
from typing import List
from collections import defaultdict

'''
TODO
Instead of removing notes from robotic MIDI if no matching note, use random noise.
Change penalty.
Test by converting mapping back to MIDI file.
'''

def getNotesMap(notes):
    '''
    Groups notes by pitch.

    Parameters
    ----------
    notes : List
    A list of notes in the form (start, pitch).

    Returns
    -------
    notes_map : dictionary
    The keys are pitches, and the values are the notes that have that pitch.
    '''
    notes_map = defaultdict(list)
    for index, (start, pitch) in enumerate(notes):
        notes_map[pitch].append((index, start))
    return notes_map

def normalise(sequence, axis=None):
    '''
    Normalises a sequence of notes such that the start time of the first note is 0,
    and the start time of the last note is 1.

    Parameters
    ----------
    sequence : np.ndarray
    A sequence of notes.
    axis : int
    The axis to normalise. Useful if you need to use a different format for sequence.

    Returns
    -------
    seqeunce : np.ndarray
    The normalised sequence of notes. The same except for start time.
    '''
    minValue = np.min(sequence[:,axis])
    sequence[:,axis] = sequence[:,axis] - minValue
    maxValue = np.max(sequence[:,axis])
    sequence[:,axis] = sequence[:,axis] / maxValue
    return sequence

def computeLCM(sequence_1, sequence_2, penalty):
    '''
    Computes the least cost of a one-to-one mapping between two sequences.
    The mapping is sequential, but NOT surjective. There may be notes that are
    excluded, if no note in the other sequence matches well enough.
    Intermediate function for retrieveLCM.

    Parameters
    ----------
    sequence_1 : np.ndarray
    A sequence of notes for a fixed pitch. Usually from the robotic MIDI.
    sequence_2 : np.ndarry
    A sequence of notes for the same pitch as sequence_1. Usually from the performance MIDI.
    penalty : int
    The penalty for not matching a note. Larger values encourage more note to be kept.

    Returns
    -------
    cost : np.ndarray
    A cost matrix. cost[i][j] is the cost of mapping sequence_1[0:i] to sequence_2[0:j].
    cost[len(sequence_1)][len(sequence_2)] is the cost of mapping the entirety of both sequences.
    NOTE : this matrix is 1-indexed. cost[0][0] is initialised to 0. 
    '''
    # list is not zero-indexed, is one-indexed
    # initialised with all ones which is good because not mapping at all is bad
    # may need to come back to this because penalty is explicit now
    cost = np.ones((len(sequence_1)+1, len(sequence_2)+1))
    # 0 to 0 should have no cost
    cost[0][0] = 0
    for i in range(1, len(sequence_1) + 1):
        for j in range(i, len(sequence_2) + 1):
            # min between ignoring jth element, mapping i to j, or ignoring ith element with some penalty
            cost[i][j] = min(cost[i][j-1], cost[i-1][j-1] + abs(sequence_1[i-1] - sequence_2[j-1]), cost[i-1][j] + penalty)
    return cost

def retrieveLCM(sequence_1, sequence_2):
    '''
    Computes the least-cost mapping from sequence_1 to sequence_2.
    Uses computeLCM to find the cost, and backtracks the computation to find
    the mapping.

    Parameters
    ----------
    sequence_1 : np.ndarray
    A sequence of notes for a fixed pitch. Usually from the robotic MIDI.
    sequence_2 : np.ndarry
    A sequence of notes for the same pitch as sequence_1. Usually from the performance MIDI.
    NOTE : penalty is not a parameter, it is hardcoded into this function. Changing it here
    will change it everywhere it's used (that's why its a paramater in computeLCM)

    Returns
    -------
    mapping : List
    A list of (sequence_1 index, sequence_2 index) pairs. These indices correspond to 
    sequence_1 and sequence_2 themselves and need to be translated (see translateMapping).
    They are also 1-indexed.
    '''
    penalty = 1/max(len(sequence_2), len(sequence_1))
    # first should always be smaller than second
    # NOTE: it makes things much easier if you just pass them in the right order.
    if len(sequence_1) > len(sequence_2):
        sequence_1, sequence_2 = sequence_2, sequence_1
    cost = computeLCM(sequence_1, sequence_2, penalty)
    i, j = len(sequence_1), len(sequence_2)
    mapping = []
    while (i > 0):
        # i is not mapped to j
        if cost[i][j] == cost[i][j-1]:
            j -= 1
        # is is mapped to j
        elif cost[i][j] == cost[i-1][j-1] + abs(sequence_1[i-1] - sequence_2[j-1]):
            mapping.append((i,j))
            i -= 1
            j -= 1
        # is is not mapped to anything
        elif cost[i][j] == cost[i-1][j] + penalty:
            # dont need to do this
            # mapping[i] = -1
            i -= 1
        else:
            raise Exception('Oh no.')
    return mapping

def translateMapping(mapping, r_notes, p_notes):
    '''
    Translates mapping to indices of the actual notes in the original sequences.

    Parameters
    ----------
    mapping : List
    A list of index pairs for a given pitch. See retrieveLCM.
    p_notes : List
    The notes for a given pitch in the performance MIDI.
    r_notes : List
    The notes for a given pitch in the robotic MIDI.

    Returns
    -------
    indices : List
    A list of index pairs for a given pitch, that correspond to the actual indices in the
    original note sequences.
    '''
    
    indices = None
    p_greater = True if len(p_notes) > len(r_notes) else False
    # If p_greater then retrieveLCM swapped the sequences around.
    if p_greater:
        indices = [(r_notes[x[0]-1][0], p_notes[x[1]-1][0]) for x in mapping]
    else:
        indices = [(r_notes[x[1]-1][0], p_notes[x[0]-1][0]) for x in mapping]
    return indices


def getMapping(performanceMIDI: pretty_midi.PrettyMIDI, roboticMIDI: pretty_midi.PrettyMIDI) -> List:
    '''
    Maps roboticMIDI to performanceMIDI. Main function.

    Parameters
    ----------
    performanceMIDI : pretty_midi.PrettyMIDI
    MIDI file corresponding to performance.
    roboticMIDI : pretty_midi.PrettyMIDI
    MIDI file corresponding to score.

    Returns
    -------
    mapping : List -> [(r_start, r_end, r_velocity, r_pitch), (p_start, p_end, p_velocity, p_pitch)]
    r_pitch is always equal to p_pitch, and the list is sorted by r_start.
    '''
    performance_notes = np.array(sorted([(note.start, note.end, note.velocity, note.pitch) for instrument in performanceMIDI.instruments for note in instrument.notes], key=lambda note : note[0]), dtype=float)
    robotic_notes = np.array(sorted([(note.start, note.end, note.velocity, note.pitch) for instrument in roboticMIDI.instruments for note in instrument.notes], key=lambda note: note[0]), dtype=float)
    performance_sequence = normalise(performance_notes[:, [0, 3]], axis=0)
    robotic_sequence = normalise(robotic_notes[:, [0, 3]], axis=0)
    performance_map = getNotesMap(performance_sequence)
    robotic_map = getNotesMap(robotic_sequence)
    indices = []
    
    for pitch in robotic_map:
        p_notes = performance_map[pitch]
        r_notes = robotic_map[pitch]
        p_seq = np.array([note[1] for note in p_notes])
        r_seq = np.array([note[1] for note in r_notes])
        mapping = retrieveLCM(r_seq, p_seq)
        mapping = translateMapping(mapping, r_notes, p_notes)
        indices += mapping
    indices.sort(key = lambda x : x[0])
    print(len(robotic_notes))
    print(len(performance_notes))
    return [(robotic_notes[x[0]], performance_notes[x[1]]) for x in indices]

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Get tempo changes between two MIDI files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('performance', action='store',
                        help='Path to the performance MIDI file.')
    parser.add_argument('robotic', action='store',
                        help='Path to the robotic MIDI file.')

    parameters = vars(parser.parse_args(sys.argv[1:]))
    print("Loading {} ...".format(parameters['performance']))
    performance = pretty_midi.PrettyMIDI(parameters['performance'])
    print("Loading {} ...".format(parameters['robotic']))
    robotic = pretty_midi.PrettyMIDI(parameters['robotic'])
    mapping = getMapping(performance, robotic)
    print(mapping)

