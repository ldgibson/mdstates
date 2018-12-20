#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <omp.h>
#include "hmm.h"


double my_timer() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return ((double)tv.tv_sec + (double)0.000001 * (double)tv.tv_usec);
}


void decode(int **obs, const int num_bonds, const int num_frames, float start_p[2], float trans_p[2][2], float emission_p[2][2], int cores) {
    omp_set_dynamic(0);
    omp_set_num_threads(cores);
    #pragma omp parallel for
    for (int i = 0; i < num_bonds; i++) {
        viterbi(obs[i], num_frames, start_p, trans_p, emission_p);
    }
}


void viterbi(int *obs, const int num_frames, float start_p[2], float trans_p[2][2], float emission_p[2][2]) {

    // 2-D array of pointers
    double **V = (double **) malloc(2 * sizeof(double *));
    double **prev = (double **) malloc(2 * sizeof(double *));

    // Declare 2nd layer of pointers
    #pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        V[i] = (double *) malloc(num_frames * sizeof(double));
        prev[i] = (double *) malloc(num_frames * sizeof(double));
    }

    // Initialize values
    #pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < num_frames; j++) {
            V[i][j] = 0;
            prev[i][j] = 0;
        }
    }

    // Get starting probabilities
    for (int st = 0; st < 2; st++) {
            V[st][0] = log10(start_p[st] * emission_p[st][obs[0]]);
    }

    // double max_prob_transfer_arr[2] = {0.0, 0.0};
    // double max_prob_transfer = 0.0;
    double max_prob = 0.0;
    int prev_st;
    double prob_0 = 0.0;
    double prob_1 = 0.0;

    // Determine probabilities starting from each point
    for (int t = 1; t < num_frames; t++) {
        for (int st = 0; st < 2; st++) {
            prob_0 = V[0][t - 1] + log10(trans_p[0][st]);
            prob_1 = V[1][t - 1] + log10(trans_p[1][st]);
            if (prob_0 > prob_1) {
                max_prob = prob_0 + log10(emission_p[st][obs[t]]);
                prev_st = 0;
            } else {
                max_prob = prob_1 + log10(emission_p[st][obs[t]]);
                prev_st = 1;
            }
            V[st][t] = max_prob;
            prev[st][t] = prev_st;

            // for (int prev_st = 0; prev_st < 2; prev_st++) {
                // max_prob_transfer_arr[prev_st] = V[t - 1][prev_st] + log10(trans_p[prev_st][st]);
            // }
            // max_prob_transfer = std::max(max_prob_transfer_arr[0], max_prob_transfer_arr[1]);
            // for (int prev_st = 0; prev_st < 2; prev_st++) {
                // if (std::abs(V[t - 1][prev_st] + log10(trans_p[prev_st][st]) - max_prob_transfer) < 0.01) {
                    // max_prob = max_prob_transfer + log10(emission_p[st][obs[t]]);
                    // V[t][st] = max_prob;
                    // prev[t][st] = prev_st;
                    // break;
                // }
            // }
        }
    }
    std::vector <int> optimal_path (num_frames, 0);
    // float max_final_prob = std::max(V[num_frames - 1][0], V[num_frames - 1][1]);
    int previous;

    // printf("mat final prob 0: %f\n", V[num_frames - 1][0]);
    // printf("mat final prob 1: %f\n", V[num_frames - 1][1]);

    // Find last optimal state
    if (V[0][num_frames - 1] > V[1][num_frames - 1]) {
        // printf("USING STATE 0\n");
        optimal_path[num_frames - 1] = 0;
        previous = 0;
    } else {
        // printf("USING STATE 1\n");
        optimal_path[num_frames - 1] = 1;
        previous = 1;
    }

    // Find last optimal state
    // for (int st = 0; st < 2; st++) {
        // if (std::abs(V[num_frames - 1][st] - max_final_prob) < 0.01) {
            // optimal_path[num_frames - 1] = st;
            // previous = st;
        // }
    // }

    // Trace back through the probabilities and find highest probabilities
    for (int t = num_frames - 2; t >= 0; t--) {
        optimal_path[t] = prev[previous][t + 1];
        previous = prev[previous][t + 1];
    }

    for (int t = 0; t < num_frames; t++) {
        obs[t] = optimal_path[t];
    }

    // return optimal_path;
}
