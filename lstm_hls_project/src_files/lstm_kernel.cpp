#include "lstm_top.hpp"
#include "hls_math.h"

static LstmStateType lstm_tanh(LstmStateType x) {
    return hls::sinh(x) / hls::cosh(x);
}

void lstm_kernel(
    const LstmWeightType weightMatrix  [INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT],
    const LstmWeightType biasVector    [TOTAL_GATE_COUNT],
    const LstmStateType  inputVector   [INPUT_FEATURE_COUNT],
          LstmStateType  hiddenState   [HIDDEN_UNIT_COUNT],
          LstmStateType  cellState     [HIDDEN_UNIT_COUNT]
) {
    #pragma HLS array_partition variable=gateInputs complete dim=1
    #pragma HLS array_partition variable=gateForget cyclic factor=64 dim=1
    #pragma HLS array_partition variable=gateForgetInv cyclic factor=64 dim=1
    #pragma HLS array_partition variable=gateInput cyclic factor=64 dim=1
    #pragma HLS array_partition variable=gateInputInv cyclic factor=64 dim=1
    #pragma HLS array_partition variable=gateOutput cyclic factor=64 dim=1
    #pragma HLS array_partition variable=gateOutputInv cyclic factor=64 dim=1
    #pragma HLS array_partition variable=cellCandidate cyclic factor=64 dim=1
    #pragma HLS array_partition variable=newCellState cyclic factor=64 dim=1

    LstmStateType gateInputs       [TOTAL_GATE_COUNT];
    LstmStateType gateInput        [HIDDEN_UNIT_COUNT];
    LstmStateType gateInputInv     [HIDDEN_UNIT_COUNT];
    LstmStateType gateForget       [HIDDEN_UNIT_COUNT];
    LstmStateType gateForgetInv    [HIDDEN_UNIT_COUNT];
    LstmStateType cellCandidate    [HIDDEN_UNIT_COUNT];
    LstmStateType gateOutput       [HIDDEN_UNIT_COUNT];
    LstmStateType gateOutputInv    [HIDDEN_UNIT_COUNT];
    LstmStateType newCellState     [HIDDEN_UNIT_COUNT];

    // Compute weighted sums + bias for all gates
    compute_weighted_sums: for (int gateIdx = 0; gateIdx < TOTAL_GATE_COUNT; ++gateIdx) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=16

        LstmStateType sum0 = biasVector[gateIdx];
        LstmStateType sum1 = 0;
        LstmStateType sum2 = 0;
        LstmStateType sum3 = 0;

        int quarterInputs = INPUT_FEATURE_COUNT / 4;
        for (int i = 0; i < quarterInputs; ++i)      sum0 += inputVector[i] * weightMatrix[i][gateIdx];
        for (int i = quarterInputs; i < 2*quarterInputs; ++i) sum1 += inputVector[i] * weightMatrix[i][gateIdx];
        for (int i = 2*quarterInputs; i < 3*quarterInputs; ++i) sum2 += inputVector[i] * weightMatrix[i][gateIdx];
        for (int i = 3*quarterInputs; i < INPUT_FEATURE_COUNT; ++i) sum3 += inputVector[i] * weightMatrix[i][gateIdx];

        gateInputs[gateIdx] = sum0 + sum1 + sum2 + sum3;
    }

    // Apply sigmoid-like transforms for input, forget, output gates
    gate_activation_loop: for (int i = 0; i < HIDDEN_UNIT_COUNT; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=32

        gateForget[i] = (LstmStateType)(1 + hls::exp((LstmStateType)-(gateInputs[i + 2*HIDDEN_UNIT_COUNT] + 1)));
        gateInput [i] = (LstmStateType)(1 + hls::exp((LstmStateType)-   gateInputs[i]));
        gateOutput[i] = (LstmStateType)(1 + hls::exp((LstmStateType)-(gateInputs[i + 3*HIDDEN_UNIT_COUNT])));
    }

    // Compute cell candidate via tanh
    cell_candidate_loop: for (int i = 0; i < HIDDEN_UNIT_COUNT; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=32

        cellCandidate[i] = lstm_tanh(gateInputs[i + HIDDEN_UNIT_COUNT]);
    }

    // Compute inverse of gate outputs
    gate_inverse_loop: for (int i = 0; i < HIDDEN_UNIT_COUNT; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=32

        gateForgetInv [i] = (LstmStateType)(1.0 / gateForget [i]);
        gateInputInv  [i] = (LstmStateType)(1.0 / gateInput  [i]);
        gateOutputInv [i] = (LstmStateType)(1.0 / gateOutput [i]);
    }

    // New cell state accumulation
    cell_update_loop: for (int i = 0; i < HIDDEN_UNIT_COUNT; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=32

        newCellState[i] = cellState[i] * gateForgetInv[i]
                        + cellCandidate[i] * gateInputInv[i];
    }

    // Apply tanh to new cell state and write back
    cell_activation_loop: for (int i = 0; i < HIDDEN_UNIT_COUNT; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=32

        cellState[i] = lstm_tanh(newCellState[i]);
    }

    // Compute updated hidden state
    hidden_update_loop: for (int i = 0; i < HIDDEN_UNIT_COUNT; ++i) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=32

        hiddenState[i] = cellState[i] * gateOutputInv[i];
    }
}
