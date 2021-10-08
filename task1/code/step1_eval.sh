#!/bin/bash

python3 disambiguator_evaluation.py \
	--model_result_path ../result/dstc10-simmc-devtest-pred-subtask-1.json \
	--data_json_path ../data/simmc2_dials_dstc10_devtest.json 
