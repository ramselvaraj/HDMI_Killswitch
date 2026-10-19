#include "tinyfacematch_ip.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool read_binary(const std::string& path, char* data, std::size_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open " << path << "\n";
        return false;
    }
    file.read(data, bytes);
    if (file.gcount() != static_cast<std::streamsize>(bytes)) {
        std::cerr << "short read from " << path << ": " << file.gcount() << " bytes\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: testbench <input.bin> <weights.bin> <expected.bin>\n";
        return 2;
    }

    const std::string input_path = argv[1];
    const std::string weights_path = argv[2];
    const std::string expected_path = argv[3];

    std::vector<uint8_t> image_bytes(TFM_IMAGE_WORDS);
    std::vector<ap_uint<8>> image(TFM_IMAGE_WORDS);
    std::vector<int16_t> weight_words(TFM_WEIGHTS_TOTAL_BYTES / 2);
    std::vector<ap_int<16>> weights(TFM_WEIGHTS_TOTAL_BYTES / 2);
    std::vector<ap_int<16>> scratch0(TFM_SCRATCH_WORDS);
    std::vector<ap_int<16>> scratch1(TFM_SCRATCH_WORDS);
    std::vector<ap_int<16>> scratch2(TFM_SCRATCH_WORDS);
    std::vector<ap_int<16>> output(TFM_EMBEDDING_DIM);
    std::vector<int16_t> expected(TFM_EMBEDDING_DIM);

    if (!read_binary(input_path, reinterpret_cast<char*>(image_bytes.data()), image_bytes.size())) {
        return 1;
    }
    if (!read_binary(weights_path, reinterpret_cast<char*>(weight_words.data()), TFM_WEIGHTS_TOTAL_BYTES)) {
        return 1;
    }
    if (!read_binary(expected_path, reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(int16_t))) {
        return 1;
    }

    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] = image_bytes[i];
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = weight_words[i];
    }

    tinyfacematch_ip(image.data(), weights.data(), scratch0.data(), scratch1.data(), scratch2.data(), output.data());

    int max_abs_error = 0;
    double mean_abs_error = 0.0;
    double dot = 0.0;
    double norm_out = 0.0;
    double norm_expected = 0.0;

    for (int i = 0; i < TFM_EMBEDDING_DIM; ++i) {
        const int out_i = static_cast<int>(output[i]);
        const int exp_i = static_cast<int>(expected[i]);
        const int error = std::abs(out_i - exp_i);
        if (error > max_abs_error) {
            max_abs_error = error;
        }
        mean_abs_error += error;
        dot += static_cast<double>(out_i) * static_cast<double>(exp_i);
        norm_out += static_cast<double>(out_i) * static_cast<double>(out_i);
        norm_expected += static_cast<double>(exp_i) * static_cast<double>(exp_i);
    }

    mean_abs_error /= (double)TFM_EMBEDDING_DIM;
    const double cosine = dot / (std::sqrt(norm_out) * std::sqrt(norm_expected) + 1e-12);

    std::cout << "max_abs_error_int16=" << max_abs_error << "\n";
    std::cout << "mean_abs_error_int16=" << mean_abs_error << "\n";
    std::cout << "cosine_to_expected=" << cosine << "\n";

    if (cosine < 0.99 || max_abs_error > 512) {
        std::cerr << "FAILED: output differs too much from expected vector\n";
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
