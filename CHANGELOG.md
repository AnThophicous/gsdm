# Changelog

All notable changes to GSDM and GVF. The project follows [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-10-25

Efficient CPU training and learning diagnostics ([ADR-0009](docs/adr/0009-efficient-cpu-training-and-learning-diagnostics.md)).

### Added
- Batched training: graphs are joined into one disjoint graph per step (`StructuralGraph.batch`,
  `GSDModel.forward_batch`), with per-graph attention readout.
- `TrainingSession`: per-epoch validation, generalization gap, precision/recall/F1 per decision, calibration
  error, best-weight tracking and early stopping.
- Resumable checkpoints (`gsdm.checkpoint`). A resumed run is bit-identical to an uninterrupted one.
- Learning diagnosis (`gsdm.diagnostics`): `learning`, `memorizing`, `stalled`, `no-validation`.
- Training logs: console lines, `metrics.jsonl` and a self-contained `dashboard.html`.
- `gsdm.capacity` and `gvf capacity`: parameters and memory of a configuration, and the hidden size for a
  target parameter count.
- GVF: `group` field in the manifest and grouped validation split; warnings for ungrouped validation and for
  categories without validation positives; `--patience`, `--checkpoint-every` and `--resume`; run directory
  `<output>.run/`.
- `benchmarks/benchmark.py`, `benchmarks/synthetic_dataset.py` and a Google Colab notebook.

### Changed
- float32 is the default dtype (`gsdm.autograd.default_dtype` switches to float64).
- Message passing uses precomputed scatter plans instead of `np.add.at`.
- `Trainer` default batch size is 32 (was 8).
- The package name is `gsdm` (the command is still `gvf`).

### Performance (Celeron N4100, same model)
- Training: 39 → 116 graphs/s. Analyzing a 2,000-line file: 868 → 637 ms, peak memory 34.5 → 15.1 MB.

### Compatibility
- `model(graph)`, `Trainer.fit` and 1.0.0 model files keep working; parameter names and shapes are unchanged.

## [1.0.0] - 2026-10-25

First public release: the GSDM architecture (from-scratch NumPy autograd, latent iterative reasoning,
per-category calibration) and GVF, the Python vulnerability finder with XML output. No pretrained model.
