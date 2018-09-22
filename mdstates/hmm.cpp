#include <stdio.h>
#include <cmath>
#include <vector>
#include "hmm.h"


float max(float x, float y) {
    if (x > y) {
        return x;
    } else {
        return y;
    }
}

std::vector<int> decoder(std::vector<int> obs) {
    int states[2] = { 0, 1 };
    float start_p[2] = { 0.5, 0.5 };
    float trans_p[2][2] = { { 0.999, 0.001 },
                            { 0.001, 0.999 } };
    float emission_p[2][2] = { { 0.60, 0.40 },
                               { 0.40, 0.60 } };


    float V[1000][2] = {{}};
    int prev[1000][2] = {{}};

    for (int st = 0; st < 2; st++) {
            V[0][st] = log10(start_p[st] * emission_p[st][obs[0]]);
    }

    float max_tr_prob[2];
    float max_tr, max_prob;
    for (int t = 1; t < 1000; t++) {
        for (int st = 0; st < 2; st++) {
            for (int prev_st = 0; prev_st < 2; prev_st++) {
                max_tr_prob[prev_st] = V[t - 1][prev_st] + log10(trans_p[prev_st][st]);
            }
            max_tr = max(max_tr_prob[0], max_tr_prob[1]);
            for (int prev_st = 0; prev_st < 2; prev_st++) {
                if (std::abs(V[t - 1][prev_st] + log10(trans_p[prev_st][st]) - max_tr) < 0.001) {
                    max_prob = max_tr + log10(emission_p[st][obs[t]]);
                    V[t][st] = max_prob;
                    prev[t][st] = prev_st;
                    break;
                }
            }
        }
    }
    std::vector <int> optimal_path;
    float max_final_prob = max(V[1000 - 1][0], V[1000 - 1][1]);
    int previous;

    for (int st = 0; st < 2; st++) {
        if (V[1000 - 1][st] == max_final_prob) {
            optimal_path.push_back(st);
            previous = st;
        }
    }

    for (int t = 1000 - 2; t >= 0; t--) {
        optimal_path.insert(optimal_path.begin(), prev[t + 1][previous]);
        previous = prev[t + 1][previous];
    }
    return optimal_path;
}
