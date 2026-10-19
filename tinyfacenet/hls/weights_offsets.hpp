#ifndef TINYFACEMATCH_HLS_WEIGHTS_HPP
#define TINYFACEMATCH_HLS_WEIGHTS_HPP

#include <cstdint>

static constexpr int TFM_WEIGHT_FRAC_BITS = 12;
static constexpr int TFM_BIAS_FRAC_BITS = 20;
static constexpr int TFM_PRELU_FRAC_BITS = 12;
static constexpr int TFM_ACT_FRAC_BITS = 10;
static constexpr int TFM_OUT_FRAC_BITS = 14;
static constexpr int TFM_WEIGHTS_TOTAL_BYTES = 258688;

static constexpr int TFM_CONV0_OFFSET_BYTES = 0;
static constexpr int TFM_DWSEP0_DW_OFFSET_BYTES = 960;
static constexpr int TFM_DWSEP0_PW_OFFSET_BYTES = 1344;
static constexpr int TFM_B0_EXPAND_OFFSET_BYTES = 2560;
static constexpr int TFM_B0_DW_OFFSET_BYTES = 7040;
static constexpr int TFM_B0_PROJECT_OFFSET_BYTES = 8576;
static constexpr int TFM_B1_EXPAND_OFFSET_BYTES = 12800;
static constexpr int TFM_B1_DW_OFFSET_BYTES = 17280;
static constexpr int TFM_B1_PROJECT_OFFSET_BYTES = 18816;
static constexpr int TFM_B2_EXPAND_OFFSET_BYTES = 23040;
static constexpr int TFM_B2_DW_OFFSET_BYTES = 27520;
static constexpr int TFM_B2_PROJECT_OFFSET_BYTES = 29056;
static constexpr int TFM_B3_EXPAND_OFFSET_BYTES = 33280;
static constexpr int TFM_B3_DW_OFFSET_BYTES = 37760;
static constexpr int TFM_B3_PROJECT_OFFSET_BYTES = 39296;
static constexpr int TFM_B4_EXPAND_OFFSET_BYTES = 47744;
static constexpr int TFM_B4_DW_OFFSET_BYTES = 64896;
static constexpr int TFM_B4_PROJECT_OFFSET_BYTES = 67968;
static constexpr int TFM_B5_EXPAND_OFFSET_BYTES = 84608;
static constexpr int TFM_B5_DW_OFFSET_BYTES = 101760;
static constexpr int TFM_B5_PROJECT_OFFSET_BYTES = 104832;
static constexpr int TFM_B6_EXPAND_OFFSET_BYTES = 121472;
static constexpr int TFM_B6_DW_OFFSET_BYTES = 138624;
static constexpr int TFM_B6_PROJECT_OFFSET_BYTES = 141696;
static constexpr int TFM_B7_EXPAND_OFFSET_BYTES = 158336;
static constexpr int TFM_B7_DW_OFFSET_BYTES = 175488;
static constexpr int TFM_B7_PROJECT_OFFSET_BYTES = 178560;
static constexpr int TFM_CONV_FINAL_OFFSET_BYTES = 195200;
static constexpr int TFM_GLOBAL_DW7_OFFSET_BYTES = 212352;
static constexpr int TFM_DENSE_OFFSET_BYTES = 225408;

#endif // TINYFACEMATCH_HLS_WEIGHTS_HPP
