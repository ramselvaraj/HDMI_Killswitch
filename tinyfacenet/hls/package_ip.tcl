set script_dir [file dirname [file normalize [info script]]]
set repo_dir [file normalize [file join $script_dir ".."]]
set input_vec [file join $repo_dir "hls" "test_vectors" "input_0.bin"]
set expected_vec [file join $repo_dir "hls" "test_vectors" "expected_0.bin"]
set weights_bin [file join $repo_dir "runs" "tinyfacematch128_w05_v4_continue" "hls_weights_q12" "weights.bin"]

open_project tinyfacematch_hls_ddr_intmac
set_top tinyfacematch_ip
add_files [file join $script_dir "tinyfacematch_ip.cpp"]
add_files -tb [file join $script_dir "testbench.cpp"]
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design -argv "$input_vec $weights_bin $expected_vec"
csynth_design
export_design -format ip_catalog -display_name "TinyFaceMatch128 DDR IntMAC" -description "TinyFaceMatch128 DDR-backed face embedding IP with integer MAC convolution kernels"
exit
