# Predicting Airline Satisfaction

Exploratory analysis and three classifiers (logistic regression, random
forest, gradient boosting) that predict whether an airline passenger was
satisfied. The work is from the Kaggle Playground Series S6E10 competition
(`playground-series-s6e10`).

The analysis lives in `predicting-airline-satisfaction.ipynb`. All the logic is
in a small package, `src/mypkg/utils.py`, which the notebook imports.

## Set up from scratch

You need [git](https://git-scm.com/downloads) and
[pixi](https://pixi.sh/latest/#installation). Pixi installs Python and every
dependency, so nothing else is needed. Windows, Linux and macOS (Intel and Apple
Silicon) are supported.

### 1. Clone the repository

```
git clone https://github.com/akhilr29/Lec_1_ex_1.git
cd Lec_1_ex_1
```

### 2. Install the environment

```
pixi install
```

This creates the environment in `.pixi/` from the exact versions in `pixi.lock`,
and installs `mypkg` into it in editable mode.

### 3. Download the data

The data is not stored in this repository. The notebook reads three CSV files
from a `data/` folder in the project root:

1. Sign in to [Kaggle](https://www.kaggle.com) and open the competition page for
   [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10).
2. Accept the competition rules, then open the **Data** tab and download the files.
3. Create a `data/` folder in the project root and put these three files in it:

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

If a file is missing, `load_data()` stops with an error that names the missing
files.

### 4. Check the install

```
pixi run pytest
```

You should see `2 passed`. These tests don't need the data.

### 5. Run the notebook

Open `predicting-airline-satisfaction.ipynb` in VS Code or JupyterLab and choose
the pixi environment as the kernel:

- **VS Code:** *Select Kernel* → *Python Environments…* → the one at
  `.pixi/envs/default` (run *Developer: Reload Window* if it isn't listed).
- **JupyterLab:** start it with `pixi run jupyter lab`.

Then run all cells in order. The random forest and gradient boosting fits use
about 700,000 rows and take a few minutes each.

## Using the package directly

```python
from mypkg.utils import load_data

df_sample, df_test, df_train = load_data()
```

To read the data from another folder, pass it in: `load_data("path/to/folder")`.

## Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mypkg'` | The notebook isn't using the pixi kernel. Select the `.pixi/envs/default` interpreter and restart the kernel. |
| `FileNotFoundError: Missing [...] in .../data` | Download the CSVs from Kaggle and put them in `data/` (step 3). |
| `pixi: command not found` | Install pixi and open a new terminal. |
