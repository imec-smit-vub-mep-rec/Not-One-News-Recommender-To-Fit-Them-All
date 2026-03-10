# 1. Prepare data (EB-NeRD)
python scripts/combine_behaviors.py --input-dir data/ebnerd/ebnerd_small

# 2. Embeddings (if CB-ST enabled)
python scripts/generate_embeddings.py --input-dir data/ebnerd/ebnerd_small --model intfloat/multilingual-e5-base

# 3. Stabilize config
python scripts/run_dataset_description.py --config config/config_ebnerd_small.json
python scripts/run_dataset_description.py --config config/config_hln.json
# Optional: feature-based outlier-removal ranking for boxplots (n=0..5)
python scripts/run_dataset_description.py --config config/config_ad.json --outlier-removal-basis total_impressions

python scripts/run_dataset_description.py --config config/config_ebnerd_small.json --outlier-removal-basis total_impressions

# 4. Update config with suggested remove_top, then run pipeline
python scripts/run_full_pipeline.py --config config/config_ebnerd_small.json --verbose
python scripts/run_full_pipeline.py --config config/config_vk_n4.json --verbose
python scripts/run_full_pipeline.py --config config/config_ad.json --verbose

# 5. (Optional) Sensitivity and plots
python scripts/run_sensitivity_analysis.py --run-dir runs/vk_20260309_165209
python scripts/plot_clusters.py --run-dir runs/ebnerd_<timestamp>

# Zip result folders
tar -czvf backup.tar.gz --exclude='*/data/*' ad_* hln_* vk_*
