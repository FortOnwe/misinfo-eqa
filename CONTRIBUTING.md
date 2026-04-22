# Contributing

Thanks for helping improve MisinfoEQA. The project is small on purpose: changes
should make dataset QA clearer, safer, or easier to reproduce.

## Local Setup

```bat
python -m pip install -e ".[full]"
python -m unittest discover -s tests
```

For dependency-light work, the demo config does not require Hugging Face access:

```bat
python -m misinfo_eqa run --config configs/demo.yaml
```

## Contribution Guidelines

- Keep baselines lightweight and interpretable unless a config explicitly opts
  into a heavier model.
- Do not add transformations that generate new misinformation. Prefer masking,
  ablation, slicing, and auditing existing examples.
- Preserve the normalized schema: `id`, `dataset`, `split`, `claim`,
  `evidence_text`, `label3`, `date`, `source_url`, and `metadata`.
- Add or update tests for label mapping, schema normalization, stressor logic,
  and CLI behavior when changing those areas.
- Treat manual-audit precision as a first-class metric. A flag that looks clever
  but wastes reviewer time should be adjusted or clearly scoped.

## Pull Request Checklist

- [ ] Unit tests pass with `python -m unittest discover -s tests`.
- [ ] New CLI behavior is documented in `README.md` or `docs/USAGE.md`.
- [ ] New stressors include a fixture or synthetic test.
- [ ] Any result claim points to a reproducible run config or artifact.
- [ ] No generated `runs/` artifacts are committed.
