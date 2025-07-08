#include "lstm_top.hpp"
#include <hls_math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <algorithm>

const std::string DATA_PATH = "/media/nisitha/My_Passport/MOODLE/Vivado_projects/LSTM_project/HLS_project/lstm_hls_project/dataset";
const int SEQUENCE_LENGTH = 7000;
const int OUTPUT_CLASS_COUNT = 6;

template<typename T, int ROWS, int COLS>
void load_csv_data(const std::string& filename, T (&array)[ROWS][COLS]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    for (int row = 0; row < ROWS; ++row) {
        std::string line;
        std::getline(file, line);
        std::stringstream iss(line);

        for (int col = 0; col < COLS; ++col) {
            std::string val;
            std::getline(iss, val, ',');
            std::stringstream convertor(val);
            double temp;
            convertor >> temp;
            array[row][col] = temp;
        }
    }
}

template<typename T, int SIZE>
void load_csv_data(const std::string& filename, T (&array)[SIZE]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    for (int i = 0; i < SIZE; ++i) {
        std::string line;
        std::getline(file, line);
        std::stringstream convertor(line);
        double temp;
        convertor >> temp;
        array[i] = temp;
    }
}

int main() {
    InputType inputSignals[SEQUENCE_LENGTH][HIDDEN_UNIT_COUNT];

    LstmWeightType dense1Weights[HIDDEN_UNIT_COUNT][HIDDEN_UNIT_COUNT];
    LstmWeightType dense1Biases[HIDDEN_UNIT_COUNT];
    LstmWeightType dense2Weights[HIDDEN_UNIT_COUNT][HIDDEN_UNIT_COUNT];
    LstmWeightType dense2Biases[HIDDEN_UNIT_COUNT];
    LstmWeightType dense3Weights[HIDDEN_UNIT_COUNT][HIDDEN_UNIT_COUNT];
    LstmWeightType dense3Biases[HIDDEN_UNIT_COUNT];

    LstmWeightType lstm1Weights[INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT];
    LstmWeightType lstm1Biases[TOTAL_GATE_COUNT];
    LstmWeightType lstm2Weights[INPUT_FEATURE_COUNT][TOTAL_GATE_COUNT];
    LstmWeightType lstm2Biases[TOTAL_GATE_COUNT];

    LstmWeightType outputWeights[HIDDEN_UNIT_COUNT][OUTPUT_CLASS_COUNT];
    LstmWeightType outputBiases[OUTPUT_CLASS_COUNT];

    int trueLabels[SEQUENCE_LENGTH];
    int predictedClasses[SEQUENCE_LENGTH];

    std::cout << "Loading test data..." << std::endl;
    load_csv_data(DATA_PATH + "data_testing.csv", inputSignals);

    load_csv_data(DATA_PATH + "weights/w_in.csv", dense1Weights);
    load_csv_data(DATA_PATH + "biases/b_in.csv", dense1Biases);
    load_csv_data(DATA_PATH + "weights/w_hidd2.csv", dense2Weights);
    load_csv_data(DATA_PATH + "biases/b_hidd2.csv", dense2Biases);
    load_csv_data(DATA_PATH + "weights/w_hidd3.csv", dense3Weights);
    load_csv_data(DATA_PATH + "biases/b_hidd3.csv", dense3Biases);

    load_csv_data(DATA_PATH + "weights_all.csv", lstm1Weights);
    load_csv_data(DATA_PATH + "biases_all.csv", lstm1Biases);
    load_csv_data(DATA_PATH + "weights_all2.csv", lstm2Weights);
    load_csv_data(DATA_PATH + "biases_all2.csv", lstm2Biases);

    load_csv_data(DATA_PATH + "weights/w_out.csv", outputWeights);
    load_csv_data(DATA_PATH + "biases/b_out.csv", outputBiases);

    load_csv_data(DATA_PATH + "labels_testing.csv", trueLabels);

    std::cout << "Running LSTM inference..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    lstm_model(
        inputSignals,
        dense1Weights, dense1Biases,
        dense2Weights, dense2Biases,
        dense3Weights, dense3Biases,
        lstm1Weights, lstm1Biases,
        lstm2Weights, lstm2Biases,
        outputWeights, outputBiases,
        predictedClasses
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int correct = 0;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        if (predictedClasses[i] == trueLabels[i]) {
            correct++;
        }
    }

    float accuracy = static_cast<float>(correct) / SEQUENCE_LENGTH * 100.0;

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Sequence Length: " << SEQUENCE_LENGTH << std::endl;
    std::cout << "Correct Predictions: " << correct << "/" << SEQUENCE_LENGTH << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Inference Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << (SEQUENCE_LENGTH * 1000.0 / duration.count()) << " samples/sec" << std::endl;

    const int NUM_SAMPLES_TO_DISPLAY = 10;
    std::cout << "\nSample Predictions (first " << NUM_SAMPLES_TO_DISPLAY << "):" << std::endl;
    std::cout << "Index\tTrue\tPredicted\tStatus" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    for (int i = 0; i < NUM_SAMPLES_TO_DISPLAY; i++) {
        std::cout << i << "\t" << trueLabels[i] << "\t" << predictedClasses[i] << "\t\t"
                  << (trueLabels[i] == predictedClasses[i] ? "✓" : "✗") << std::endl;
    }

    if (accuracy > 70.0) {
        std::cout << "\nTEST PASSED (Accuracy > 70%)" << std::endl;
        return 0;
    } else {
        std::cout << "\nTEST FAILED (Accuracy <= 70%)" << std::endl;
        return -1;
    }
}
