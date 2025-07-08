############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project lstm_hls_project
set_top top_fpga_lstm
add_files lstm_hls_project/src_files/lstm_kernel.cpp
add_files lstm_hls_project/src_files/lstm_model.cpp
add_files lstm_hls_project/src_files/lstm_top.hpp
add_files -tb lstm_hls_project/src_files/tb_lstm.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7z010clg400-1}
create_clock -period 10 -name default
#source "./lstm_hls_project/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
