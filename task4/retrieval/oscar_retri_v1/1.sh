python3 oscar/run_objectid.py \
    --model_name_or_path   pretrained/pretrained_base/checkpoint-2000000 \
    --eval_model_dir output/checkpoint-1-770  \
    --do_eval \
    --evaluate_during_training \
    --data_dir 'data' 
