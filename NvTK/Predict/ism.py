import itertools
import numpy as np

def in_silico_mutagenesis_sequences(sequence,
                                    mutate_n_bases=1,
                                    reference_sequence=Genome,
                                    start_position=0,
                                    end_position=None):
    if end_position is None:
        end_position = len(sequence)
    if start_position >= end_position:
        raise ValueError(("Starting positions must be less than the ending "
                          "positions. Found a starting position of {0} with "
                          "an ending position of {1}.").format(start_position,
                                                               end_position))
    if start_position < 0:
        raise ValueError("Negative starting positions are not supported.")
    if end_position < 0:
        raise ValueError("Negative ending positions are not supported.")
    if start_position >= len(sequence):
        raise ValueError(("Starting positions must be less than the sequence length."
                          " Found a starting position of {0} with a sequence length "
                          "of {1}.").format(start_position, len(sequence)))
    if end_position > len(sequence):
        raise ValueError(("Ending positions must be less than or equal to the sequence "
                          "length. Found an ending position of {0} with a sequence "
                          "length of {1}.").format(end_position, len(sequence)))
    if (end_position - start_position) < mutate_n_bases:
        raise ValueError(("Fewer bases exist in the substring specified by the starting "
                          "and ending positions than need to be mutated. There are only "
                          "{0} currently, but {1} bases must be mutated at a "
                          "time").format(end_position - start_position, mutate_n_bases))

    sequence_alts = []
    for index, ref in enumerate(sequence):
        alts = []
        for base in reference_sequence.BASES_ARR:
            if base == ref:
                continue
            alts.append(base)
        sequence_alts.append(alts)
    all_mutated_sequences = []
    for indices in itertools.combinations(
            range(start_position, end_position), mutate_n_bases):
        pos_mutations = []
        for i in indices:
            pos_mutations.append(sequence_alts[i])
        for mutations in itertools.product(*pos_mutations):
            all_mutated_sequences.append(list(zip(indices, mutations)))
    return all_mutated_sequences


def mutate_sequence(encoding,
                    mutation_information,
                    reference_sequence=Genome):
    mutated_seq = np.copy(encoding)
    for (position, alt) in mutation_information:
        replace_base = reference_sequence.BASE_TO_INDEX[alt]
        mutated_seq[position, :] = 0
        mutated_seq[position, replace_base] = 1
    return mutated_seq


def _ism_sample_id(sequence, mutation_information):
    """
    TODO

    Parameters
    ----------
    sequence : str
        The input sequence to mutate.
    mutation_information : list(tuple)
        TODO

    Returns
    -------
    TODO
        TODO

    """
    positions = []
    refs = []
    alts = []
    for (position, alt) in mutation_information:
        positions.append(str(position))
        refs.append(sequence[position])
        alts.append(alt)
    return (';'.join(positions), ';'.join(refs), ';'.join(alts))
