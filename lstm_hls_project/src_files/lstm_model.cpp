#include "lstm_top.hpp"
#include <hls_math.h>
#include <ap_int.h>

#define SEQUENCE_LENGTH     7000
#define OUTPUT_CLASS_COUNT  6

using InputType = ap_fixed<9,9>;

void lstm_kernel(
    const LstmWeightType weightMatrix   [INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT],
    const LstmWeightType biasVector     [TOTAL_GATE_COUNT],
    const LstmStateType  inputVector    [INPUT_FEATURE_COUNT],
          LstmStateType  hiddenState    [HIDDEN_UNIT_COUNT],
          LstmStateType  cellState      [HIDDEN_UNIT_COUNT]
);

void lstm_model(
    InputType            inputSignals    [SEQUENCE_LENGTH][HIDDEN_UNIT_COUNT],
    LstmWeightType       dense1Weights   [HIDDEN_UNIT_COUNT][HIDDEN_UNIT_COUNT],
    LstmWeightType       dense1Biases    [HIDDEN_UNIT_COUNT],
    LstmWeightType       dense2Weights   [HIDDEN_UNIT_COUNT][HIDDEN_UNIT_COUNT],
    LstmWeightType       dense2Biases    [HIDDEN_UNIT_COUNT],
    LstmWeightType       dense3Weights   [HIDDEN_UNIT_COUNT][HIDDEN_UNIT_COUNT],
    LstmWeightType       dense3Biases    [HIDDEN_UNIT_COUNT],
    LstmWeightType       lstm1Weights    [INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT],
    LstmWeightType       lstm1Biases     [TOTAL_GATE_COUNT],
    LstmWeightType       lstm2Weights    [INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT],
    LstmWeightType       lstm2Biases     [TOTAL_GATE_COUNT],
    LstmWeightType       outputWeights   [HIDDEN_UNIT_COUNT][OUTPUT_CLASS_COUNT],
    LstmWeightType       outputBiases    [OUTPUT_CLASS_COUNT],
    int                  outputClasses   [SEQUENCE_LENGTH]
) {
    LstmStateType dense1Out [SEQUENCE_LENGTH][HIDDEN_UNIT_COUNT];
    LstmStateType dense2Out [SEQUENCE_LENGTH][HIDDEN_UNIT_COUNT];
    LstmStateType logits    [SEQUENCE_LENGTH][OUTPUT_CLASS_COUNT];

    LstmStateType hidden1[HIDDEN_UNIT_COUNT] = {0};
    LstmStateType cell1  [HIDDEN_UNIT_COUNT] = {0};
    LstmStateType hidden2[HIDDEN_UNIT_COUNT] = {0};
    LstmStateType cell2  [HIDDEN_UNIT_COUNT] = {0};

    LstmStateType lstmInput1[INPUT_FEATURE_COUNT];
    LstmStateType lstmInput2[INPUT_FEATURE_COUNT];

	#pragma HLS array_partition variable = inputSignals cyclic factor=64 dim=2
	#pragma HLS array_partition variable = dense1Out cyclic factor=64 dim=2
	#pragma HLS array_partition variable = dense2Out cyclic factor=64 dim=2
	#pragma HLS array_partition variable = logits cyclic factor=6 dim=2

    // Dense Layer 1 with Sigmoid Activation
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
        for (int j = 0; j < HIDDEN_UNIT_COUNT; ++j) {
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=16
            LstmStateType acc = dense1Biases[j];
            for (int k = 0; k < HIDDEN_UNIT_COUNT; ++k)
                acc += inputSignals[t][k] * dense1Weights[k][j];
            dense1Out[t][j] = 1 / (1 + hls::exp(-acc));
        }
    }

    // Dense Layer 2 (Linear)
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
        for (int j = 0; j < HIDDEN_UNIT_COUNT; ++j) {
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=16
            LstmStateType acc = dense2Biases[j];
            for (int k = 0; k < HIDDEN_UNIT_COUNT; ++k)
                acc += dense1Out[t][k] * dense2Weights[k][j];
            dense2Out[t][j] = acc;
        }
    }

    // Dense Layer 3 (Linear)
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
        for (int j = 0; j < HIDDEN_UNIT_COUNT; ++j) {
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=16
            LstmStateType acc = dense3Biases[j];
            for (int k = 0; k < HIDDEN_UNIT_COUNT; ++k)
                acc += dense2Out[t][k] * dense3Weights[k][j];
            dense1Out[t][j] = acc; // reuse dense1Out as new activations
        }
    }

    // LSTM Layer 1 + LSTM Layer 2
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
        for (int j = 0; j < HIDDEN_UNIT_COUNT; ++j) {
			#pragma HLS unroll factor=64
            lstmInput1[j]             = dense1Out[t][j];
            lstmInput1[j + HIDDEN_UNIT_COUNT] = hidden1[j];
        }
        lstm_kernel(lstm1Weights, lstm1Biases, lstmInput1, hidden1, cell1);

        for (int j = 0; j < HIDDEN_UNIT_COUNT; ++j) {
			#pragma HLS unroll factor=64
            lstmInput2[j]             = hidden1[j];
            lstmInput2[j + HIDDEN_UNIT_COUNT] = hidden2[j];
        }
        lstm_kernel(lstm2Weights, lstm2Biases, lstmInput2, hidden2, cell2);

        for (int j = 0; j < HIDDEN_UNIT_COUNT; ++j) {
			#pragma HLS unroll factor=64
            dense2Out[t][j] = hidden2[j];
        }
    }

    // Output Layer (Dense)
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
        for (int j = 0; j < OUTPUT_CLASS_COUNT; ++j) {
			#pragma HLS pipeline II=1
            LstmStateType acc = outputBiases[j];
            for (int k = 0; k < HIDDEN_UNIT_COUNT; ++k)
                acc += dense2Out[t][k] * outputWeights[k][j];
            logits[t][j] = acc;
        }
    }

    // Argmax (Classification)
    for (int t = 0; t < SEQUENCE_LENGTH; ++t) {
        LstmStateType maxLogit = logits[t][0];
        int maxIndex = 0;
        for (int j = 1; j < OUTPUT_CLASS_COUNT; ++j) {
            if (logits[t][j] > maxLogit) {
                maxLogit = logits[t][j];
                maxIndex = j;
            }
        }
        outputClasses[t] = maxIndex;
    }
}
