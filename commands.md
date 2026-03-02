## Draft commands for quick reference while running the pipeline
Note: I will remove this document once the pipeline is stable and all the results are in.

```bash
python scripts/run_full_pipeline.py --config config/config_ad.json --verbose

python scripts/run_full_pipeline.py --config config/config_ebnerd_small.json --verbose
python scripts/run_full_pipeline.py --config config/config_hln.json --verbose

python scripts/run_full_pipeline.py --config config/config_vk.json --verbose

python scripts/run_full_pipeline.py --config config/config_vk.json

python scripts/plot_clusters.py --run-dir runs/vk_20260225_203509/data

python scripts/plot_clusters.py --run-dir runs/hln_20260226_104643/data --log-x
```