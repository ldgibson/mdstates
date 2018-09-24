#include <stdio.h>
#include <cmath>
#include <vector>
#include "hmm.h"


std::vector<int> decoder(std::vector<int> obs, const int num_frames,
                         std::vector<float> start_p,
                         std::vector<std::vector<float>> trans_p,
                         std::vector<std::vector<float>> emission_p) {
    // float start_p[2] = { 0.5, 0.5 };
    // float trans_p[2][2] = { { 0.999, 0.001 },
                            // { 0.001, 0.999 } };
    // float emission_p[2][2] = { { 0.60, 0.40 },
                               // { 0.40, 0.60 } };

    // float start_p[2] = {start_p_val, float(1.0 - start_p_val)};
    // float trans_p[2][2];
    // trans_p[0][0] = trans_p_val;
    // trans_p[1][1] = trans_p_val;
    // trans_p[0][1] = 1.0 - trans_p_val;
    // trans_p[1][0] = 1.0 - trans_p_val;

    // float emission_p[2][2];
    // emission_p[0][0] = emission_p_val;
    // emission_p[1][1] = emission_p_val;
    // emission_p[0][1] = 1.0 - emission_p_val;
    // emission_p[1][0] = 1.0 - emission_p_val;

    float(*V)[2] = new float[num_frames][2];
    int(*prev)[2] = new int[num_frames][2];

    for (int st = 0; st < 2; st++) {
            V[0][st] = log10(start_p[st] * emission_p[st][obs[0]]);
    }

    float max_tr_prob[2];
    float max_tr, max_prob;
    for (int t = 1; t < num_frames; t++) {
        for (int st = 0; st < 2; st++) {
            for (int prev_st = 0; prev_st < 2; prev_st++) {
                max_tr_prob[prev_st] = V[t - 1][prev_st] + log10(trans_p[prev_st][st]);
            }
            max_tr = std::max(max_tr_prob[0], max_tr_prob[1]);
            for (int prev_st = 0; prev_st < 2; prev_st++) {
                if (std::abs(V[t - 1][prev_st] + log10(trans_p[prev_st][st]) - max_tr) < 0.0001) {
                    max_prob = max_tr + log10(emission_p[st][obs[t]]);
                    V[t][st] = max_prob;
                    prev[t][st] = prev_st;
                    break;
                }
            }
        }
    }
    std::vector <int> optimal_path;
    float max_final_prob = std::max(V[num_frames - 1][0], V[num_frames - 1][1]);
    int previous;

    for (int st = 0; st < 2; st++) {
        if (V[num_frames - 1][st] == max_final_prob) {
            optimal_path.push_back(st);
            previous = st;
        }
    }

    for (int t = num_frames - 2; t >= 0; t--) {
        optimal_path.insert(optimal_path.begin(), prev[t + 1][previous]);
        previous = prev[t + 1][previous];
    }
    delete [] V;
    delete [] prev;
    return optimal_path;
}
