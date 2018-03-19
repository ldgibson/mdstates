import numpy as np


def viterbi(obs, states, start_p, trans_p, emission_p):
    """Algorithm used to decode a signal and find the optimal path.

    Viterbi algorithm used to decode a noisy trajectory"""

    V = [{}]
    for st in states:
        V[0][st] = {"log_prob": np.log10(start_p[st] * emission_p[st, obs[0]]),
                    "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["log_prob"] +
                              np.log10(trans_p[prev_st, st])
                              for prev_st in states)

            for prev_st in states:
                if V[t-1][prev_st]["log_prob"] + \
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
        optimal_path.insert(0, V[t+1][previous]["prev"])
        previous = V[t+1][previous]["prev"]

    return optimal_path
