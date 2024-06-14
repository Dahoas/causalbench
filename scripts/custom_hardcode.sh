method=$1
echo $method
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory ~/repos/bio_agent/causalbench/output/retrieval_llm_llama \
    --data_directory ~/repos/bio_agent/causalbench/data \
    --training_regime observational \
    --model_name custom \
    --inference_function_file_path causalscbench/models/llm.py \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter \
    --max_path_length -1 \
    --omission_estimation_size 500 \
    --path_to_network_csv ~/repos/bio_agent/causalbench/output/retrieval_llm/658392/output_network.csv