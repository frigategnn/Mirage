#!/bin/bash
if [ $# -lt 1 ]
then
    printf "usage: %s <GPU-ID>\n" "$0"
    exit 1
fi
gpu=$1
epochs=100
runs=10
dataset_name="DD"
dataset_path="dataset/pyg_DD.pkl"
params_file="configs.DD"
params_obj_name="DD_gin_config"
output_dir="outputs"
exp_num_file="exp.txt"
if [ ! -e "${exp_num_file}" ]
then
    echo 0 > "${exp_num_file}"
fi
run_num=$(head -1 "${exp_num_file}" | cut -d' ' -f1 | tr -d '\n')
run_num=$((run_num+1))
echo "${run_num}" > "${exp_num_file}"
if [ $# -eq 1 ]
then
    CUDA_VISIBLE_DEVICES="${gpu}" python3 ldDDrminetrees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --dataset_path "${dataset_path}" --output_dir "${output_dir}" --runs "${runs}"
else
    if [ $# -eq 2 ]
    then
        if [ $2 = "--debug" ]
        then
            CUDA_VISIBLE_DEVICES="${gpu}" python3 -m pdb ldDDrminetrees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --dataset_path "${dataset_path}" --output_dir "${output_dir}" --runs "${runs}"
        else
            CUDA_VISIBLE_DEVICES="${gpu}" python3 ldDDrminetrees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --dataset_path "${dataset_path}" --output_dir "${output_dir}" --runs "${runs}"
        fi
    else
        CUDA_VISIBLE_DEVICES="${gpu}" python3 ldDDrminetrees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --dataset_path "${dataset_path}" --output_dir "${output_dir}" --runs "${runs}"
    fi
fi
