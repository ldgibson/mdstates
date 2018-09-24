import numpy as np

from .hmm_cython import decoder_cpp

__all__ = ['generate_ignore_list', 'viterbi', 'viterbi_wrapper']


def generate_ignore_list(cmat, n):
    """
    Generates an ignore list of indices.

    For a given matrix `cmat`, find all indices that do not ever change
    more than `n` times and record their most common value.

    Parameters
    ----------
    cmat : numpy.ndarray
        Contact matrix.
    n : int
        Threshold for being placed on the ignore list.

    Returns
    -------
    ignore_list : list of list of int
        All indices in `ignore_list[0]` do not express a value other
        than 0 frequently enough to be processed. The same anology goes
        for `ignore_list[1]`.
    """
    ignore_list = [[], []]
    n_atoms = cmat.shape[1]

    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            # Find all unique values and their respective counts
            values, counts = np.unique(cmat[:, i, j], return_counts=True)
            unique = dict(zip(values, counts))

            # If 0 or 1 never appear, place that index on
            # the ignore_list.
            if 1 not in unique.keys():
                ignore_list[0].append([i, j])
                continue
            elif 0 not in unique.keys():
                ignore_list[1].append([i, j])
                continue
            else:
                pass

            if unique[1] <= n:
                ignore_list[0].append([i, j])
            elif unique[0] <= n:
                ignore_list[1].append([i, j])
            else:
                pass
    return ignore_list


def viterbi(obs, states, start_p, trans_p, emission_p):
    """Algorithm used to decode a signal and find the optimal path.

    Parameters
    ----------
    obs : numpy.ndarray
        Numerical representations of the observations.
    states : list of int
        Numerical representations of the states in the system.
    start_p : list of float
        Probabilities of starting in a particular hidden state.
    trans_p : list of list of float
        Probabilities of transitioning from one hidden state to
        another.
    emission_p : list of of list of float
        Probabilities of emitting an observable given the present
        hidden state.

    Returns
    -------
    optimal_path : list of int
        Optimal path of the system given the observations and Hidden
        Markov Model parameters."""

    V = [{}]
    for st in states:
        V[0][st] = {"log_prob": np.log10(start_p[st] * emission_p[st, obs[0]]),
                    "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t - 1][prev_st]["log_prob"] +
                              np.log10(trans_p[prev_st, st])
                              for prev_st in states)

            for prev_st in states:
                if V[t - 1][prev_st]["log_prob"] + \
                        np.log10(trans_p[prev_st, st]) == max_tr_prob:
                    max_prob = max_tr_prob + np.log10(emission_p[st, obs[t]])
                    V[t][st] = {"log_prob": max_prob, "prev": prev_st}
                    break

    optimal_path = []

    # The highest probability
    max_final_prob = max(value["log_prob"] for value in V[-1].values())
    previous = None

    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["log_prob"] == max_final_prob:
            optimal_path.append(st)
            previous = st
            break

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        optimal_path.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return optimal_path


def viterbi_wrapper(inp_params):
    # states = np.array([0, 1])
    start_p = np.array([0.5, 0.5])
    trans_p = np.array([[0.999, 0.001], [0.001, 0.999]])
    emission_p = np.array([[0.6, 0.4], [0.4, 0.6]])

    i = inp_params[0]
    j = inp_params[1]
    obs = inp_params[2]

    return (i, j, decoder_cpp(obs, start_p, trans_p, emission_p))


def fast_viterbi(obs, states, start_p, trans_p, emission_p):
    """Algorithm used to decode a signal and find the optimal path.

    Parameters
    ----------
    obs : numpy.ndarray
        Numerical representations of the observations.
    states : list of int
        Numerical representations of the states in the system.
    start_p : list of float
        Probabilities of starting in a particular hidden state.
    trans_p : list of list of float
        Probabilities of transitioning from one hidden state to
        another.
    emission_p : list of list of float
        Probabilities of emitting an observable given the present
        hidden state.

    Returns
    -------
    optimal_path : list of int
        Optimal path of the system given the observations and Hidden
        Markov Model parameters."""

    V = np.zeros((len(obs), len(states)))
    prev_states = np.zeros((len(obs), len(states)), dtype=int)

    V[0, :] = np.log10(start_p[:] * emission_p[:, obs[0]])

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        for st in states:
            max_tr_prob = (V[t - 1, :] + np.log10(trans_p[:, st])).max()
            V[t, st] = max_tr_prob + np.log10(emission_p[st, obs[t]])
            prev_states[t, st] = np.where(V[t - 1, :] +
                                          np.log10(trans_p[:, st]) ==
                                          max_tr_prob)[0][0]

    optimal_path = np.zeros(len(obs), dtype=int)

    # The highest final probability.
    max_final_prob = V[-1, :].max()

    # Get most probable state and its backtrack.
    optimal_path[-1] = np.where(V[-1, :] == max_final_prob)[0][0]
    previous = optimal_path[-1]

    # Follow the each backtrack from the saved paths.
    for t in range(len(V) - 2, -1, -1):
        optimal_path[t] = prev_states[t + 1, previous]
        previous = optimal_path[t]

    return optimal_path
