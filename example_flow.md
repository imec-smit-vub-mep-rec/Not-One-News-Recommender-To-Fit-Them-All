# 1. Prepare data (EB-NeRD)
python scripts/combine_behaviors.py --input-dir data/ebnerd/ebnerd_small

# 2. Embeddings (if CB-ST enabled)
python scripts/generate_embeddings.py --input-dir data/ebnerd/ebnerd_small --model intfloat/multilingual-e5-base

# 3. Stabilize config
python scripts/run_dataset_description.py --config config/config_ebnerd_small.json
python scripts/run_dataset_description.py --config config/config_hln.json

# 4. Update config with suggested remove_top, then run pipeline
python scripts/run_full_pipeline.py --config config/config_ebnerd_small.json --verbose
python scripts/run_full_pipeline.py --config config/config_vk_n4.json --verbose

# 5. (Optional) Sensitivity and plots
python scripts/run_sensitivity_analysis.py --run-dir runs/ebnerd_<timestamp>
python scripts/plot_clusters.py --run-dir runs/ebnerd_<timestamp>