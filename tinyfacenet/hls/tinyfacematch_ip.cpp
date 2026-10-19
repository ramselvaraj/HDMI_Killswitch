#include "tinyfacematch_ip.hpp"

#include <hls_math.h>

static constexpr int TFM_MAX_CHANNELS = 128;
static constexpr int TFM_MAX_DW_KERNEL_WORDS = 49;
static constexpr int TFM_CONV0_KERNEL_WORDS = 3 * 3 * 3;
static constexpr int TFM_ACC_FRAC_BITS = TFM_ACT_FRAC_BITS + TFM_WEIGHT_FRAC_BITS;

static ap_int<32> read_i32_words(const ap_int<16>* words, int index) {
#pragma HLS INLINE
    ap_uint<16> lo = (ap_uint<16>)words[index];
    ap_uint<16> hi = (ap_uint<16>)words[index + 1];
    ap_uint<32> combined = ((ap_uint<32>)hi << 16) | lo;
    return (ap_int<32>)combined;
}

static tfm_act_t dequant_i16_to_act(ap_int<16> word, int frac_bits) {
#pragma HLS INLINE
    tfm_wide_t scale = (tfm_wide_t)1.0 / (tfm_wide_t)(1 << frac_bits);
    return (tfm_act_t)((tfm_wide_t)word * scale);
}

static tfm_acc_t dequant_i32_to_acc(ap_int<32> word, int frac_bits) {
#pragma HLS INLINE
    tfm_wide_t scale = (tfm_wide_t)1.0 / (tfm_wide_t)(1 << frac_bits);
    return (tfm_acc_t)((tfm_wide_t)word * scale);
}

static ap_int<16> quantize_act(tfm_act_t value) {
#pragma HLS INLINE
    tfm_wide_t scaled = (tfm_wide_t)value * (tfm_wide_t)(1 << TFM_ACT_FRAC_BITS);
    if (scaled >= 0) {
        scaled += (tfm_wide_t)0.5;
    } else {
        scaled -= (tfm_wide_t)0.5;
    }
    if (scaled > 32767) {
        return 32767;
    }
    if (scaled < -32768) {
        return -32768;
    }
    return (ap_int<16>)scaled;
}

static ap_int<16> quantize_output(tfm_acc_t value) {
#pragma HLS INLINE
    tfm_wide_t scaled = (tfm_wide_t)value * (tfm_wide_t)(1 << TFM_OUT_FRAC_BITS);
    if (scaled >= 0) {
        scaled += (tfm_wide_t)0.5;
    } else {
        scaled -= (tfm_wide_t)0.5;
    }
    if (scaled > 32767) {
        return 32767;
    }
    if (scaled < -32768) {
        return -32768;
    }
    return (ap_int<16>)scaled;
}

static tfm_act_t read_act(const ap_int<16>* tensor, int c, int y, int x, int h, int w) {
#pragma HLS INLINE
    int idx = (c * h + y) * w + x;
    return dequant_i16_to_act(tensor[idx], TFM_ACT_FRAC_BITS);
}

static ap_int<16> read_act_raw(const ap_int<16>* tensor, int c, int y, int x, int h, int w) {
#pragma HLS INLINE
    int idx = (c * h + y) * w + x;
    return tensor[idx];
}

static ap_int<16> read_image_q10(const ap_uint<8>* image_rgb, int c, int y, int x) {
#pragma HLS INLINE
    int idx = (y * 112 + x) * 3 + c;
    int pixel = (int)image_rgb[idx];
    return (ap_int<16>)(pixel * 8 - 1020);
}

static ap_int<16> saturate_i16(ap_int<48> value) {
#pragma HLS INLINE
    if (value > 32767) {
        return 32767;
    }
    if (value < -32768) {
        return -32768;
    }
    return (ap_int<16>)value;
}

static ap_int<16> round_shift_sat_i16(ap_int<48> value, int shift) {
#pragma HLS INLINE
    ap_int<48> half = (ap_int<48>)1 << (shift - 1);
    ap_int<48> shifted;
    if (value >= 0) {
        shifted = (value + half) >> shift;
    } else {
        shifted = -(((-value) + half) >> shift);
    }
    return saturate_i16(shifted);
}

static ap_int<16> apply_prelu_q10(ap_int<16> value, ap_int<16> alpha_q12) {
#pragma HLS INLINE
    if (value >= 0) {
        return value;
    }
    ap_int<48> product = (ap_int<48>)value * (ap_int<48>)alpha_q12;
    return round_shift_sat_i16(product, TFM_PRELU_FRAC_BITS);
}

static ap_int<16> add_q10_sat(ap_int<16> a, ap_int<16> b) {
#pragma HLS INLINE
    return saturate_i16((ap_int<48>)a + (ap_int<48>)b);
}

static void run_conv_image(
    const ap_uint<8>* image_rgb,
    ap_int<16>* output,
    const ap_int<16>* weights,
    int offset_words
) {
#pragma HLS INLINE off
    const int weight_offset = offset_words;
    const int bias_offset = weight_offset + 16 * 3 * 3 * 3;
    const int prelu_offset = bias_offset + 16 * 2;

    for (int oc = 0; oc < 16; ++oc) {
        ap_int<16> w_cache[TFM_CONV0_KERNEL_WORDS];
        for (int i = 0; i < TFM_CONV0_KERNEL_WORDS; ++i) {
#pragma HLS PIPELINE II=1
            w_cache[i] = weights[weight_offset + oc * TFM_CONV0_KERNEL_WORDS + i];
        }
        ap_int<48> bias = (ap_int<48>)read_i32_words(weights, bias_offset + oc * 2) << (TFM_ACC_FRAC_BITS - TFM_BIAS_FRAC_BITS);
        ap_int<16> alpha = weights[prelu_offset + oc];

        for (int oy = 0; oy < 56; ++oy) {
            for (int ox = 0; ox < 56; ++ox) {
                ap_int<48> acc = bias;
                for (int ic = 0; ic < 3; ++ic) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            int iy = oy * 2 + ky - 1;
                            int ix = ox * 2 + kx - 1;
                            if (iy >= 0 && iy < 112 && ix >= 0 && ix < 112) {
                                int widx = (ic * 3 + ky) * 3 + kx;
                                acc += (ap_int<48>)read_image_q10(image_rgb, ic, iy, ix) * (ap_int<48>)w_cache[widx];
                            }
                        }
                    }
                }
                ap_int<16> out = round_shift_sat_i16(acc, TFM_WEIGHT_FRAC_BITS);
                output[(oc * 56 + oy) * 56 + ox] = apply_prelu_q10(out, alpha);
            }
        }
    }
}

static void run_conv1x1(
    const ap_int<16>* input,
    ap_int<16>* output,
    const ap_int<16>* residual,
    const ap_int<16>* weights,
    int in_c,
    int out_c,
    int in_h,
    int in_w,
    int offset_words,
    bool has_prelu,
    bool add_residual
) {
#pragma HLS INLINE off
    int out_h = in_h;
    int out_w = in_w;
    int weight_offset = offset_words;
    int bias_offset = weight_offset + out_c * in_c;
    int prelu_offset = bias_offset + out_c * 2;

    for (int oc = 0; oc < out_c; oc += 2) {
        ap_int<16> w_cache0[TFM_MAX_CHANNELS];
        ap_int<16> w_cache1[TFM_MAX_CHANNELS];
        bool has_oc1 = (oc + 1) < out_c;
        for (int ic = 0; ic < in_c; ++ic) {
#pragma HLS PIPELINE II=1
            w_cache0[ic] = weights[weight_offset + oc * in_c + ic];
            if (has_oc1) {
                w_cache1[ic] = weights[weight_offset + (oc + 1) * in_c + ic];
            }
        }
        ap_int<48> bias0 = (ap_int<48>)read_i32_words(weights, bias_offset + oc * 2) << (TFM_ACC_FRAC_BITS - TFM_BIAS_FRAC_BITS);
        ap_int<48> bias1 = 0;
        if (has_oc1) {
            bias1 = (ap_int<48>)read_i32_words(weights, bias_offset + (oc + 1) * 2) << (TFM_ACC_FRAC_BITS - TFM_BIAS_FRAC_BITS);
        }
        ap_int<16> alpha0 = 0;
        ap_int<16> alpha1 = 0;
        if (has_prelu) {
            alpha0 = weights[prelu_offset + oc];
            if (has_oc1) {
                alpha1 = weights[prelu_offset + oc + 1];
            }
        }

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                ap_int<48> acc0 = bias0;
                ap_int<48> acc1 = bias1;
                for (int ic = 0; ic < in_c; ++ic) {
                    ap_int<16> in_value = read_act_raw(input, ic, oy, ox, in_h, in_w);
                    acc0 += (ap_int<48>)in_value * (ap_int<48>)w_cache0[ic];
                    if (has_oc1) {
                        acc1 += (ap_int<48>)in_value * (ap_int<48>)w_cache1[ic];
                    }
                }
                ap_int<16> out0 = round_shift_sat_i16(acc0, TFM_WEIGHT_FRAC_BITS);
                if (has_prelu) {
                    out0 = apply_prelu_q10(out0, alpha0);
                }
                if (add_residual) {
                    out0 = add_q10_sat(out0, read_act_raw(residual, oc, oy, ox, out_h, out_w));
                }
                output[(oc * out_h + oy) * out_w + ox] = out0;

                if (has_oc1) {
                    ap_int<16> out1 = round_shift_sat_i16(acc1, TFM_WEIGHT_FRAC_BITS);
                    if (has_prelu) {
                        out1 = apply_prelu_q10(out1, alpha1);
                    }
                    if (add_residual) {
                        out1 = add_q10_sat(out1, read_act_raw(residual, oc + 1, oy, ox, out_h, out_w));
                    }
                    output[((oc + 1) * out_h + oy) * out_w + ox] = out1;
                }
            }
        }
    }
}

static void run_depthwise(
    const ap_int<16>* input,
    ap_int<16>* output,
    const ap_int<16>* weights,
    int channels,
    int in_h,
    int in_w,
    int k,
    int stride,
    int pad,
    int offset_words,
    bool has_prelu
) {
#pragma HLS INLINE off
    int out_h = (in_h + 2 * pad - k) / stride + 1;
    int out_w = (in_w + 2 * pad - k) / stride + 1;
    int weight_offset = offset_words;
    int bias_offset = weight_offset + channels * k * k;
    int prelu_offset = bias_offset + channels * 2;

    for (int c = 0; c < channels; ++c) {
        ap_int<16> w_cache[TFM_MAX_DW_KERNEL_WORDS];
        for (int i = 0; i < k * k; ++i) {
#pragma HLS PIPELINE II=1
            w_cache[i] = weights[weight_offset + c * k * k + i];
        }
        ap_int<48> bias = (ap_int<48>)read_i32_words(weights, bias_offset + c * 2) << (TFM_ACC_FRAC_BITS - TFM_BIAS_FRAC_BITS);
        ap_int<16> alpha = 0;
        if (has_prelu) {
            alpha = weights[prelu_offset + c];
        }

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                ap_int<48> acc = bias;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        int iy = oy * stride + ky - pad;
                        int ix = ox * stride + kx - pad;
                        if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                            int widx = ky * k + kx;
                            acc += (ap_int<48>)read_act_raw(input, c, iy, ix, in_h, in_w) * (ap_int<48>)w_cache[widx];
                        }
                    }
                }
                ap_int<16> out = round_shift_sat_i16(acc, TFM_WEIGHT_FRAC_BITS);
                if (has_prelu) {
                    out = apply_prelu_q10(out, alpha);
                }
                output[(c * out_h + oy) * out_w + ox] = out;
            }
        }
    }
}

static void run_dense_l2(
    const ap_int<16>* input,
    ap_int<16>* embedding_out,
    const ap_int<16>* weights,
    int offset_words
) {
#pragma HLS INLINE off
    tfm_act_t dense_out[128];
    tfm_act_t input_cache[128];
    int weight_offset = offset_words;
    int bias_offset = weight_offset + 128 * 128;

    for (int i = 0; i < 128; ++i) {
#pragma HLS PIPELINE II=1
        input_cache[i] = read_act(input, i, 0, 0, 1, 1);
    }

    tfm_acc_t sum = 0;
    for (int o = 0; o < 128; ++o) {
        tfm_acc_t acc = dequant_i32_to_acc(read_i32_words(weights, bias_offset + o * 2), TFM_BIAS_FRAC_BITS);
        for (int i = 0; i < 128; ++i) {
            int widx = weight_offset + o * 128 + i;
            tfm_act_t w = dequant_i16_to_act(weights[widx], TFM_WEIGHT_FRAC_BITS);
            acc += (tfm_acc_t)input_cache[i] * (tfm_acc_t)w;
        }
        dense_out[o] = (tfm_act_t)acc;
        sum += (tfm_acc_t)dense_out[o] * (tfm_acc_t)dense_out[o];
    }

    tfm_acc_t inv_norm = (tfm_acc_t)1.0 / hls::sqrt(sum + (tfm_acc_t)0.000001);
    for (int i = 0; i < 128; ++i) {
        tfm_acc_t normalized = (tfm_acc_t)dense_out[i] * inv_norm;
        embedding_out[i] = quantize_output(normalized);
    }
}

void tinyfacematch_ip(
    const ap_uint<8>* image_rgb,
    const ap_int<16>* weights,
    ap_int<16>* scratch0,
    ap_int<16>* scratch1,
    ap_int<16>* scratch2,
    ap_int<16>* embedding_out
) {
#pragma HLS INTERFACE m_axi port=image_rgb offset=slave bundle=gmem0 depth=37632
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1 depth=129344
#pragma HLS INTERFACE m_axi port=scratch0 offset=slave bundle=gmem2 depth=200704
#pragma HLS INTERFACE m_axi port=scratch1 offset=slave bundle=gmem2 depth=200704
#pragma HLS INTERFACE m_axi port=scratch2 offset=slave bundle=gmem2 depth=200704
#pragma HLS INTERFACE m_axi port=embedding_out offset=slave bundle=gmem2 depth=128
#pragma HLS INTERFACE s_axilite port=image_rgb bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=scratch0 bundle=control
#pragma HLS INTERFACE s_axilite port=scratch1 bundle=control
#pragma HLS INTERFACE s_axilite port=scratch2 bundle=control
#pragma HLS INTERFACE s_axilite port=embedding_out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    run_conv_image(image_rgb, scratch0, weights, TFM_CONV0_OFFSET_BYTES / 2);
    run_depthwise(scratch0, scratch1, weights, 16, 56, 56, 3, 1, 1, TFM_DWSEP0_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch1, scratch0, scratch0, weights, 16, 32, 56, 56, TFM_DWSEP0_PW_OFFSET_BYTES / 2, true, false);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 32, 64, 56, 56, TFM_B0_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 64, 56, 56, 3, 2, 1, TFM_B0_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 64, 32, 28, 28, TFM_B0_PROJECT_OFFSET_BYTES / 2, false, false);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 32, 64, 28, 28, TFM_B1_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 64, 28, 28, 3, 1, 1, TFM_B1_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 64, 32, 28, 28, TFM_B1_PROJECT_OFFSET_BYTES / 2, false, true);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 32, 64, 28, 28, TFM_B2_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 64, 28, 28, 3, 1, 1, TFM_B2_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 64, 32, 28, 28, TFM_B2_PROJECT_OFFSET_BYTES / 2, false, true);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 32, 64, 28, 28, TFM_B3_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 64, 28, 28, 3, 2, 1, TFM_B3_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 64, 64, 14, 14, TFM_B3_PROJECT_OFFSET_BYTES / 2, false, false);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 64, 128, 14, 14, TFM_B4_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 128, 14, 14, 3, 1, 1, TFM_B4_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 128, 64, 14, 14, TFM_B4_PROJECT_OFFSET_BYTES / 2, false, true);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 64, 128, 14, 14, TFM_B5_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 128, 14, 14, 3, 1, 1, TFM_B5_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 128, 64, 14, 14, TFM_B5_PROJECT_OFFSET_BYTES / 2, false, true);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 64, 128, 14, 14, TFM_B6_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 128, 14, 14, 3, 2, 1, TFM_B6_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 128, 64, 7, 7, TFM_B6_PROJECT_OFFSET_BYTES / 2, false, false);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 64, 128, 7, 7, TFM_B7_EXPAND_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch2, weights, 128, 7, 7, 3, 1, 1, TFM_B7_DW_OFFSET_BYTES / 2, true);
    run_conv1x1(scratch2, scratch0, scratch0, weights, 128, 64, 7, 7, TFM_B7_PROJECT_OFFSET_BYTES / 2, false, true);

    run_conv1x1(scratch0, scratch1, scratch0, weights, 64, 128, 7, 7, TFM_CONV_FINAL_OFFSET_BYTES / 2, true, false);
    run_depthwise(scratch1, scratch0, weights, 128, 7, 7, 7, 1, 0, TFM_GLOBAL_DW7_OFFSET_BYTES / 2, false);
    run_dense_l2(scratch0, embedding_out, weights, TFM_DENSE_OFFSET_BYTES / 2);
}
