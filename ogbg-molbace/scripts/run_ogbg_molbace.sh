#!/bin/bash
if [ $# -lt 1 ]
then
    printf "usage: %s <GPU-ID>\n" "$0"
    exit 1
fi
gpu=$1
epochs=30
runs=10
dataset_name="ogbg-molbace"
params_file="configs.molbace"
params_obj_name="molbace_gat_config"
exp_file="exp.txt"
output_dir="outputs"
if [ ! -e "${exp_file}" ]
then
    echo 0 > "${exp_file}"
fi
run_num=$(head -1 "${exp_file}" | cut -d' ' -f1 | tr -d '\n')
run_num=$((run_num+1))
echo "${run_num}" > "${exp_file}"
if [ $# -eq 1 ]
then
    CUDA_VISIBLE_DEVICES="${gpu}" python3 mine_trees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --output_dir "${output_dir}" --runs "${runs}"
else
    if [ $# -eq 2 ]
    then
        if [ $2 = "--debug" ]
        then
            CUDA_VISIBLE_DEVICES="${gpu}" python3 -m pdb mine_trees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --output_dir "${output_dir}" --runs "${runs}"
        else
            CUDA_VISIBLE_DEVICES="${gpu}" python3 mine_trees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --output_dir "${output_dir}" --runs "${runs}"
        fi
    else
        CUDA_VISIBLE_DEVICES="${gpu}" python3 mine_trees.py --epochs "${epochs}" --run_num "${run_num}" --params_file "${params_file}" --params_obj_name "${params_obj_name}" --dataset_name "${dataset_name}" --output_dir "${output_dir}" --runs "${runs}"
    fi
fi
