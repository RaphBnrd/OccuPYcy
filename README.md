<p align="center">
  <img src="assets/logo_cropped.png" height="120" alt="OccuPYcy logo">
</p>

**OccuPYcy** is an open-source <u>**PY**</u>thon package for site-<u>**occu**</u>pan<u>**cy**</u> modeling at scale. OccuPYcy provides tools for flexible, high-performance fitting of single or multi-species occupancy models using maximum likelihood with (stochastic) gradient-based optimization in [PyTorch](https://pytorch.org/). It is designed for scalability, deep model expressivity, and robust application to massive, uneven datasets—such as those from citizen science.

## Features

- **Flexible Model Definitions:** Use linear, non-linear, or neural networks for occupancy and detection probabilities via PyTorch modules.
- **Extensible:** Define custom architectures and features for modern site-occupancy modeling.
- **Supports Large, Heterogeneous Datasets:** Memory mapping and efficient batching for millions of observations with highly uneven visits per site. Suited for citizen science data with variable sampling and detection effort.
- **Multiple Optimizers:** LBFGS, SGD, Adam, and other first- and second-order optimizers.
- **Penalized Estimation:** Easily apply L1/L2 regularization and other penalties.
- **Open Source:** code available on GitHub under the MIT License, soon on PyPI. *A scientific publication describing and applying the method is in preparation.*

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for reproducible local development.  
*Python ≥3.8 and PyTorch ≥2.0 required.*

### Quick local setup:

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
```

If not using `uv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
```

## Quick Start Example


## Site-Occupancy Framework on Citizen Science Data


<p align="center">
  <img src="assets/opportunistic_PO_data-diagram.png" width="80%" alt="Illustration of citizen science data bias">
</p>

<p align="center">
  <img src="assets/occupancy-diagram.png" width="60%" alt="Illustration of occupancy modeling framework">
</p>

## How to Cite

If you use OccuPYcy in your research, please cite the following reference:

```
Benerradi, R., Botella, C., Servajean, M., & Joly, A. (2024). "Increasing the Expressivity and Scalability of Site-Occupancy Modeling for Species Trend Estimation from Citizen Science Data." *Manuscript in preparation*. [GitHub Repository](https://github.com/RaphBnrd/Plantnet-Trend-Estimation)
```

BibTeX:
```bibtex
@article{benerradi2024occupycy,
  title = {Increasing the Expressivity and Scalability of Site-Occupancy Modeling for Species Trend Estimation from Citizen Science Data},
  author = {Benerradi, Raphaël and Botella, Christophe and Servajean, Maximilien and Joly, Alexis},
  note = {Manuscript in preparation. \url{https://github.com/RaphBnrd/Plantnet-Trend-Estimation}},
  year = {2024}
}
```

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/RaphBnrd/Plantnet-Trend-Estimation/issues) or submit a pull request. See `CONTRIBUTING.md` for details.

## License

OccuPYcy is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Project Links

- [Source code on GitHub](https://github.com/RaphBnrd/Plantnet-Trend-Estimation)
- [Example usage](examples/)
- For questions, contact [raphael.benerradi@inria.fr](mailto:raphael.benerradi@inria.fr)

---

*OccuPYcy was developed as part of ongoing research in scalable machine learning for ecology.*