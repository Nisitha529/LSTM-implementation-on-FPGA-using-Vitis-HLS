// lstm_top.hpp
#ifndef LSTM_TOP_HPP
#define LSTM_TOP_HPP

#include <ap_fixed.h>

// — Configuration Parameters —
constexpr int INPUT_FEATURE_COUNT = 128;
constexpr int HIDDEN_UNIT_COUNT   = 64;
constexpr int TOTAL_GATE_COUNT    = 4 * HIDDEN_UNIT_COUNT;

// — Fixed‑Point Types —
using LstmWeightType = ap_fixed<14, 2>;
using LstmStateType  = ap_fixed<14, 6>;

// — LSTM Kernel Declaration —
void lstm_kernel(
    const LstmWeightType weightMatrix   [INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT],
    const LstmWeightType biasVector     [TOTAL_GATE_COUNT],
    const LstmStateType  inputVector    [INPUT_FEATURE_COUNT],
          LstmStateType  hiddenState    [HIDDEN_UNIT_COUNT],
          LstmStateType  cellState      [HIDDEN_UNIT_COUNT]
);

#endif // LSTM_TOP_HPP
