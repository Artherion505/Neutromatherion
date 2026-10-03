# Neutromatherion: Paper Code

This repository contains selected source code and tests associated with the Neutromatherion paper. It is a work in progress, not a complete or observationally validated cosmological solver. In particular, the Planck script plots a released reference spectrum; it does not calculate a Neutromatherion CMB prediction.

## Code map

- `flrw_*.py`, `background_data_fit.py`, and `neutromatherion_cmb.py`: background and perturbation calculations.
- `sparc_analysis.py`: SPARC rotation-curve analysis and figure.
- `solar_thinshell.py`, `solar_radial_validation.py`, and `plot_scan.py`: solar-screening calculations and figures.
- `planck_tt_check.py`: Planck TT reference-data check and figure.
- `test_*.py` and `solartest.py`: automated checks and test helpers.

## Reproduce checks

Use Python 3.10 or compatible. Install dependencies with `python -m pip install -r requirements-paper.txt`.

The core tests do not require external datasets:

```powershell
python -m unittest -q test_flrw_coupled test_neutromatherion_cmb
```

This command passes 119 tests in the reference environment (2026-10-03). The tests check implemented code paths; they do not establish observational validation of the complete framework.

The background-fit and solar-profile tests require files listed in `DATA_SOURCES.md`:

```powershell
python -m unittest test_background_data_fit test_solar_thinshell
```

The plotting scripts also require their corresponding external input datasets. No raw datasets or generated plots are included in this repository.

## Citation and credit

If this framework or its central ideas inform later work, please acknowledge **David López-Fernández** and cite this repository. When a versioned Zenodo record is available, please cite that record and its DOI. `CITATION.cff` provides GitHub's citation metadata.

Si este marco o sus ideas centrales sirven de base para trabajo posterior, se agradece mencionar a **David López-Fernández** y citar este repositorio. Cuando haya una versión publicada en Zenodo, cita ese registro y su DOI.

## Access and licenses

Anyone can inspect, clone, or fork this public repository. Changes to the canonical `main` branch require repository write access. Python source files are provided under the MIT terms in `LICENSE-MIT.txt`; documentation is under CC BY 4.0. External datasets retain their own terms; see `DATA_SOURCES.md` before obtaining or redistributing them.