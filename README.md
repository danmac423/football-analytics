# Football Analytics

### Program do analizy meczów piłki nożnej
Projekt polega na stworzeniu programu służącemu analizie meczów piłki nożnej na podstawie nagrań wideo. Umożliwi on użytkownikowi wczytanie nagrania meczu i wygeneruje analizę, która obejmuje:

- Oznaczenie zawodników i bramkarzy z rozróżnieniem na drużyny oraz sędziów na nagraniu,
- Wyświetlanie prędkości poszczególnych piłkarzy,
- Wyświetlanie statystyki posiadania piłki przez drużyny,
- Wyświetlanie radaru boiska z zaznaczonymi pozycjami piłkarzy.

### Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── DESIGN_PROPOSAL.md <- The project info and schedule
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         football_analytics and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── football_analytics   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes football_analytics a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

### Autorzy

- Dominika Boguszewska
- Daniel Machniak
- Natalia Pieczko