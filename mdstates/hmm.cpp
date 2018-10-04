#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include "hmm.h"


std::vector<int> decoder(std::vector<int> obs, const int num_frames,
                         float start_p[2], float trans_p[2][2], float emission_p[2][2]) {
    // 2-D array of pointers
    double **V = (double **) malloc(num_frames * sizeof(double *));
    double **prev = (double **) malloc(num_frames * sizeof(double *));

    // Declare 2nd layer of pointers
    for (int i = 0; i < num_frames; i++) {
        V[i] = (double *) malloc(2 * sizeof(double));
        prev[i] = (double *) malloc(2 * sizeof(double));
    }

    // Initialize values
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < 2; j++) {
            V[i][j] = 0;
            prev[i][j] = 0;
        }
    }

    // Get starting probabilities
    for (int st = 0; st < 2; st++) {
            V[0][st] = log10(start_p[st] * emission_p[st][obs[0]]);
    }

    double max_prob_transfer_arr[2] = {0.0, 0.0};
    double max_prob_transfer = 0.0;
    double max_prob = 0.0;

    // Determine probabilities starting from each point
    for (int t = 1; t < num_frames; t++) {
        for (int st = 0; st < 2; st++) {
            for (int prev_st = 0; prev_st < 2; prev_st++) {
                max_prob_transfer_arr[prev_st] = V[t - 1][prev_st] + log10(trans_p[prev_st][st]);
            }
            max_prob_transfer = std::max(max_prob_transfer_arr[0], max_prob_transfer_arr[1]);
            for (int prev_st = 0; prev_st < 2; prev_st++) {
                if (std::abs(V[t - 1][prev_st] + log10(trans_p[prev_st][st]) - max_prob_transfer) < 0.0001) {
                    max_prob = max_prob_transfer + log10(emission_p[st][obs[t]]);
                    V[t][st] = max_prob;
                    prev[t][st] = prev_st;
                    break;
                }
            }
        }
    }
    std::vector <int> optimal_path (num_frames, 0);
    float max_final_prob = std::max(V[num_frames - 1][0], V[num_frames - 1][1]);
    int previous;

    // Find last optimal state
    for (int st = 0; st < 2; st++) {
        if (V[num_frames - 1][st] == max_final_prob) {
            optimal_path[num_frames - 1] = st;
            previous = st;
        }
    }

    // Trace back through the probabilities and find highest probabilities
    for (int t = num_frames - 2; t >= 0; t--) {
        optimal_path[t] = prev[t + 1][previous];
        previous = prev[t + 1][previous];
    }
    return optimal_path;
}
