#!/bin/bash

methods=("lasso" "random_forest" "grnboost" "genie" "ges" "gies" "pc" "mvpc" "gsp" "igsp" "notears-lin" "notears-lin-sparse" "notears-mlp" "notears-mlp-sparse" "DCDI-G" "DCDI-DSF" "DCDFG-LIN" "DCDFG-MLP" "corum" "lr" "string_network" "string_physical" "custom" "chipseq" "pooled_biological_networks" "sortnregress")

for method in "${methods[@]}"; do
    bash scripts/default_run.sh $method
done