# SD2.1 + PixArt Base/UGILE Code Package

This folder contains zip archives of the active code used for SD2.1 and PixArt base-vs-UGILE smoke runs.

Included:
- `sd2.1/` active source, config, and prompt YAML files.
- `PIXART/` active source, config, and prompt YAML files.
- `tools/run_ugile_smoke_tests.py`
- `tools/ugile_pair_manifest.py`
- `tools/verify_ugile_pairs.py`
- `tools/check_ugile_mandatory_consistency.py`
- `tools/ugile_prompt_utils.py`
- `environment_versions.txt`
- `requirements_diverse_lock.txt`
- Smoke-run configs from `outputs/ugile_smoke/_configs/SD21_smoke_config.yaml` and `PIXART_smoke_config.yaml`.

Excluded:
- Generated images and logs.
- Python bytecode caches.
- External model weights and Hugging Face cache files.

Current active UGILE mode:
- SD2.1: threshold UGILE, `theta_max=0.35`, `noise_scale=2.0`
- PixArt: threshold UGILE, `theta_max=2.0`, `noise_scale=15.0`

Typical smoke command after extracting inside the repository root:

```bash
/data1/sakib/miniconda3/envs/diverse/bin/python -u tools/run_ugile_smoke_tests.py --mode pair --gpu 0 --arch SD21 --seed 41
```

```bash
/data1/sakib/miniconda3/envs/diverse/bin/python -u tools/run_ugile_smoke_tests.py --mode pair --gpu 0 --arch PIXART --seed 41
```
