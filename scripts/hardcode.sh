method=$1
echo $method
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory ~/repos/bio_agent/causalbench/output/mannwhitney \
    --data_directory ~/repos/bio_agent/causalbench/data \
    --training_regime interventional \
    --model_name custom \
    --inference_function_file_path causalscbench/models/mannwhitney.py \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter \
    --max_path_length -1 \
    --omission_estimation_size 500 \
    --path_to_network_csv ~/repos/bio_agent/causalbench/output/ges/606334/output_network.csv