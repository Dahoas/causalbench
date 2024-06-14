method=$1
echo $method
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory ~/repos/bio_agent/causalbench/output/$method \
    --data_directory ~/repos/bio_agent/causalbench/data \
    --training_regime observational \
    --model_name $method \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter \
    --max_path_length -1 \
    --omission_estimation_size 500