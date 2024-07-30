#!/bin/bash
M_array=(1 8 16 32 64 128 256 512 1024 2048 4096)
K_array=(4096 4096 11008)
N_array=(4096 12288 4096)

# Set GRP SIZE TO 32
sed -i "s/static int const K_SUBV = .*;/static int const K_SUBV = 32;/g" config.h 
sed -i "s/static int const GRP_SIZE = .*;/static int const GRP_SIZE = 32;/g" config.h
sed -i "s/K_SUBV = .*;/K_SUBV = 32;/g" aiegraph/bf16_gemm_bds.py
sed -i "s/GRP_SIZE = .*;/GRP_SIZE = 32;/g" aiegraph/bf16_gemm_bds.py 
mkdir -p "./gemm_shapes/grp_32/"
sed -i "s/bf16_gemm_bds.*.py/bf16_gemm_bds.py/g"  Makefile 
for M_dim in ${M_array[@]}
do
    for i in "${!K_array[@]}";
    do
        echo "Running GEMM for: $M_dim x ${K_array[i]} x ${N_array[i]}"
        sed -i "s/int const Mgemm = .*;/int const Mgemm = $M_dim;/g" 4x4/main.cc
        sed -i "s/Mgemm = .*/Mgemm = $M_dim;/g" aiegraph/bf16_gemm_bds.py 
        sed -i "s/int const Kgemm = .*;/int const Kgemm = ${K_array[i]};/g" 4x4/main.cc
        sed -i "s/Kgemm = .*/Kgemm = ${K_array[i]};/g" aiegraph/bf16_gemm_bds.py 
        sed -i "s/int const Ngemm = .*;/int const Ngemm = ${N_array[i]};/g" 4x4/main.cc
        sed -i "s/Ngemm = .*/Ngemm = ${N_array[i]};/g" aiegraph/bf16_gemm_bds.py 
        make all 4x4=1
        make txn_hw
        mkdir -p "./gemm_shapes/grp_32/${M_dim}_${K_array[i]}_${N_array[i]}"
        cp -rf hw_package "./gemm_shapes/grp_32/${M_dim}_${K_array[i]}_${N_array[i]}/." 
    done
done

# Set GRP SIZE TO 128
sed -i "s/static int const K_SUBV = .*;/static int const K_SUBV = 128;/g" config.h 
sed -i "s/static int const GRP_SIZE = .*;/static int const GRP_SIZE = 128;/g" config.h
sed -i "s/K_SUBV = .*;/K_SUBV = 128;/g" aiegraph/bf16_gemm_bds.py
sed -i "s/GRP_SIZE = .*;/GRP_SIZE = 128;/g" aiegraph/bf16_gemm_bds.py 
mkdir -p "./gemm_shapes/grp_128/"
for M_dim in ${M_array[@]}
do
    for i in "${!K_array[@]}";
    do
        echo "Running GEMM for: $M_dim x ${K_array[i]} x ${N_array[i]}"
        sed -i "s/int const Mgemm = .*;/int const Mgemm = $M_dim;/g" 4x4/main.cc
        sed -i "s/Mgemm = .*/Mgemm = $M_dim;/g" aiegraph/bf16_gemm_bds.py 
        sed -i "s/int const Kgemm = .*;/int const Kgemm = ${K_array[i]};/g" 4x4/main.cc
        sed -i "s/Kgemm = .*/Kgemm = ${K_array[i]};/g" aiegraph/bf16_gemm_bds.py 
        sed -i "s/int const Ngemm = .*;/int const Ngemm = ${N_array[i]};/g" 4x4/main.cc
        sed -i "s/Ngemm = .*/Ngemm = ${N_array[i]};/g" aiegraph/bf16_gemm_bds.py 
        make all 4x4=1
        make txn_hw
        mkdir -p "./gemm_shapes/grp_128/${M_dim}_${K_array[i]}_${N_array[i]}"
        cp -rf hw_package "./gemm_shapes/grp_128/${M_dim}_${K_array[i]}_${N_array[i]}/." 
    done
done

#N=32768 is generated with a different python DMA compiler
M_array=(1 8 32)
K_array=(4096)
N_array=(32768)

# Set GRP SIZE TO 32
sed -i "s/bf16_gemm_bds.*.py/bf16_gemm_bds_N_32K.py/g"  Makefile 
sed -i "s/static int const K_SUBV = .*;/static int const K_SUBV = 32;/g" config.h 
sed -i "s/static int const GRP_SIZE = .*;/static int const GRP_SIZE = 32;/g" config.h
sed -i "s/K_SUBV = .*;/K_SUBV = 32;/g" aiegraph/bf16_gemm_bds_N_32K.py
sed -i "s/GRP_SIZE = .*;/GRP_SIZE = 32;/g" aiegraph/bf16_gemm_bds_N_32K.py 
for M_dim in ${M_array[@]}
do
    for i in "${!K_array[@]}";
    do
        echo "Running GEMM for: $M_dim x ${K_array[i]} x ${N_array[i]}"
        sed -i "s/int const Mgemm = .*;/int const Mgemm = $M_dim;/g" 4x4/main.cc
        sed -i "s/Mgemm = .*/Mgemm = $M_dim;/g" aiegraph/bf16_gemm_bds_N_32K.py 
        sed -i "s/int const Kgemm = .*;/int const Kgemm = ${K_array[i]};/g" 4x4/main.cc
        sed -i "s/Kgemm = .*/Kgemm = ${K_array[i]};/g" aiegraph/bf16_gemm_bds_N_32K.py 
        sed -i "s/int const Ngemm = .*;/int const Ngemm = ${N_array[i]};/g" 4x4/main.cc
        sed -i "s/Ngemm = .*/Ngemm = ${N_array[i]};/g" aiegraph/bf16_gemm_bds_N_32K.py 
        make all 4x4=1
        make txn_hw
        mkdir -p "./gemm_shapes/grp_32/${M_dim}_${K_array[i]}_${N_array[i]}"
        cp -rf hw_package "./gemm_shapes/grp_32/${M_dim}_${K_array[i]}_${N_array[i]}/." 
    done
done

# Set GRP SIZE TO 128
sed -i "s/static int const K_SUBV = .*;/static int const K_SUBV = 128;/g" config.h 
sed -i "s/static int const GRP_SIZE = .*;/static int const GRP_SIZE = 128;/g" config.h
sed -i "s/K_SUBV = .*;/K_SUBV = 128;/g" aiegraph/bf16_gemm_bds_N_32K.py
sed -i "s/GRP_SIZE = .*;/GRP_SIZE = 128;/g" aiegraph/bf16_gemm_bds_N_32K.py 
for M_dim in ${M_array[@]}
do
    for i in "${!K_array[@]}";
    do
        echo "Running GEMM for: $M_dim x ${K_array[i]} x ${N_array[i]}"
        sed -i "s/int const Mgemm = .*;/int const Mgemm = $M_dim;/g" 4x4/main.cc
        sed -i "s/Mgemm = .*/Mgemm = $M_dim;/g" aiegraph/bf16_gemm_bds_N_32K.py 
        sed -i "s/int const Kgemm = .*;/int const Kgemm = ${K_array[i]};/g" 4x4/main.cc
        sed -i "s/Kgemm = .*/Kgemm = ${K_array[i]};/g" aiegraph/bf16_gemm_bds_N_32K.py 
        sed -i "s/int const Ngemm = .*;/int const Ngemm = ${N_array[i]};/g" 4x4/main.cc
        sed -i "s/Ngemm = .*/Ngemm = ${N_array[i]};/g" aiegraph/bf16_gemm_bds_N_32K.py 
        make all 4x4=1
        make txn_hw
        mkdir -p "./gemm_shapes/grp_128/${M_dim}_${K_array[i]}_${N_array[i]}"
        cp -rf hw_package "./gemm_shapes/grp_128/${M_dim}_${K_array[i]}_${N_array[i]}/." 
    done
done
