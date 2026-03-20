# Epidemic Modelling Code

This repository is used to explore how infection spread changes with rewiring on a two-dimensional network.

It contains three key files:

- `sirs_network_model.py`
- `analysis_functions.py`
- `main.py`

as well as two additional files, `well_mixed_SIR.py` and `covid_data.py`, which did not contribute to the final report but were used more heavily in the earlier stages of the project before the final direction was fixed.

---

## Core Files

### `sirs_network_model.py`

This is the core file of the project.

It defines the `Networks` class, which provides the main functionality required to:

- generate two-dimensional small-world networks with rewiring probability \(p\)
- simulate stochastic SIR and SIRS dynamics on these networks
- track infection quantities over time, including cumulative infections
- generate spatial snapshots and infection time series
- analyse additional behaviour such as long-range branching and oscillations

All simulations and measurements in the project are built on top of this file.

---

### `analysis_functions.py`

This file contains supporting analysis functions used throughout the project.

Its purpose is to keep the main script shorter and clearer by placing repeated analysis tasks into separate functions. These include:

- epidemic threshold estimation
- model fitting for threshold curves
- peak infection and peak time calculations
- cumulative infection plotting
- oscillation analysis
- secondary-seed infection measurements

These functions work on top of the simulation methods defined in `sirs_network_model.py`.

---

### `main.py`

This is the main script used to run the project.

It combines the network model and helper functions to produce the analyses and figures used in the report. This includes:

- generating example simulations and visualisations
- computing epidemic thresholds across rewiring probabilities
- analysing peak infection behaviour
- plotting cumulative infection dynamics
- measuring endemic oscillations
- running additional appendix-style analyses

Whilst the code in `main.py` will reproduce the results in the report up to potential random seeding effects, some sections were not run exactly in this single form due to computational expense and were instead executed in smaller chunks.

---

## Other Files

### `well_mixed_SIR.py`

This file contains an exploratory two-strain SIR/SIRS-style model under the well-mixed assumption.

It was used in earlier exploratory work to investigate how two competing strains might spread through a population, including cases where one strain is introduced after another and where cross-immunity is asymmetric. Although this did not contribute to the final report, it was kept as part of the wider project.

---

### `covid_data.py`

This file contains a utility class for reading and plotting COVID data from ONS spreadsheets.

It was used in early exploratory work as a possible route for comparing epidemic behaviour in the model with real infection data. Although it was not used in the final report, it was retained as part of the project codebase.

---

Run with:

```bash
python main.py