# stable-worldmodel Development

Development repository for [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), a platform for reproducible world model research and evaluation.

## Quick Start

### Installation

```bash
pip install 'stable-worldmodel[all]'
```

### Sample Environment

See [`sample_environment.py`](sample_environment.py) for a complete example demonstrating the three stages of world model research:

1. **Data Collection** — Collect demonstration data from an environment
2. **Training** — Load the dataset and train a world model
3. **Evaluation** — Evaluate with model-predictive control

```bash
python sample_environment.py
```

## References

- [Official Repository](https://github.com/galilai-group/stable-worldmodel)
- [Documentation](https://galilai-group.github.io/stable-worldmodel/)
- [Paper](https://arxiv.org/abs/2605.21800)

## Citation

```bibtex
@misc{maes_lld2026swm,
  title={stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation},
  author={Lucas Maes and Quentin Le Lidec and Luiz Facury and Nassim Massaudi and Ayush Chaurasia and Francesco Capuano and Richard Gao and Taj Gillin and Dan Haramati and Damien Scieur and Yann LeCun and Randall Balestriero},
  year={2026},
  eprint={2605.21800},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.21800},
}
```
