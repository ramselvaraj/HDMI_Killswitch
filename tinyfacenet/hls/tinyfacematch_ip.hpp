#ifndef TINYFACEMATCH_DDR_IP_HPP
#define TINYFACEMATCH_DDR_IP_HPP

#include <ap_fixed.h>
#include <ap_int.h>

#include "weights_offsets.hpp"

static constexpr int TFM_IMAGE_WORDS = 112 * 112 * 3;
static constexpr int TFM_SCRATCH_WORDS = 64 * 56 * 56;
static constexpr int TFM_EMBEDDING_DIM = 128;

typedef ap_fixed<16, 6, AP_RND, AP_SAT> tfm_act_t;
typedef ap_fixed<32, 14, AP_RND, AP_SAT> tfm_acc_t;
typedef ap_fixed<64, 32, AP_RND, AP_SAT> tfm_wide_t;

void tinyfacematch_ip(
    const ap_uint<8>* image_rgb,
    const ap_int<16>* weights,
    ap_int<16>* scratch0,
    ap_int<16>* scratch1,
    ap_int<16>* scratch2,
    ap_int<16>* embedding_out
);

#endif // TINYFACEMATCH_DDR_IP_HPP
