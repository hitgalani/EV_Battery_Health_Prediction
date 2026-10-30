# EV Battery Health Prediction: Project Report

**Author:** Hit Galani
**Repository:** https://github.com/hitgalani/EV_Battery_Health_Prediction
**Project type:** Beginner supervised machine-learning learning project
**Environment:** Windows, PowerShell, VS Code, Python virtual environment

> **Safety and data warning:** The dataset is synthetic. This project is for learning only. It is not a real battery inspection, certified diagnosis, or battery safety tool. The course asks for real-world data, so get teacher approval before submitting this dataset.

> **How this report was made:** It was written from the recorded project context (README details and earlier run records). The Python source files (`project.py`, `main.py`, `main2.py`) were **not inspected** and the code was **not re-run** while writing this report. Anything marked **[VERIFY]** must be checked against your actual code before you present it. No new scores or predictions were created.

---

## 1. Title and Project Overview

This project uses five EV battery measurements to:

1. Predict numeric **State of Health (SoH, %)** using **Linear Regression**.
2. Predict the dataset's **battery status** category using **Logistic Regression**.

It walks through a full beginner machine-learning workflow: load data, clean it, explore it, split it, train models, evaluate them, and predict from user input.

## 2. Introduction and Problem Statement

**State of Health (SoH)** describes how much a battery has aged compared with a new one. 100% means like new; lower values mean more wear.

**Problem:** Can five measurements in the supplied dataset help estimate SoH and predict the status label recorded in that dataset?

Because the dataset is synthetic, the answer only describes this dataset. It does not show how well the models work on real batteries.

## 3. Project Objectives

1. Inspect and prepare the supplied CSV.
2. Explore its values with tables and charts.
3. Train simple and multiple Linear Regression models for SoH.
4. Train Logistic Regression for the existing battery-status label.
5. Evaluate models with a held-out test set and five-fold cross-validation.
6. Ask the user for five measurements and show an educational prediction.

## 4. Project Folders and Important Files

Expected structure from the README:

```text
EV_Battery_Health_Prediction/
|-- Dataset/
|   `-- EV_Battery_Dataset.csv
|-- Python_Code/
|   `-- project.py
|-- Reports/
|   |-- EV_Battery_Project_Report.html
|   |-- Sweetviz_EDA_Report.html
|   `-- *.png
|-- Learning_Website/
|   |-- index.html
|   `-- images/
|-- README.md
|-- requirements.txt
`-- .gitignore
```

| Item | Purpose |
|---|---|
| `Dataset/EV_Battery_Dataset.csv` | The data used for training and testing |
| `Python_Code/project.py` | The main script the README says to run |
| `Reports/` | Saved charts, result tables, and the Sweetviz HTML report |
| `Learning_Website/` | Offline beginner guide with images |
| `README.md` | Project description and instructions |
| `requirements.txt` | Exact library versions |
| `.gitignore` | Files Git should ignore |

**Extra items seen in the local VS Code workspace [VERIFY]:** `.venv/` (virtual environment), `.vscode/` (editor settings), `Fundamentals of Machine Learning/` (course material), `Main/` (with `Main-2-Reports/` and `Main-Reports/`), `main.py`, and `main2.py`. The README does not describe `main.py`, `main2.py` or `Main/`. They may be alternate or newer versions of `project.py`. Open them and compare before explaining their purpose. The README also links `PROJECT_GUIDE.md`, which was not visible in the workspace.

## 5. Dataset

| Detail | Value |
|---|---|
| Name | Electric Vehicle (EV) Battery Degradation and Charge |
| Source | [Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) |
| File | `Dataset/EV_Battery_Dataset.csv` |
| Recorded size | 10,000 rows x 13 columns |
| Data type | Synthetic |

| # | Column | Meaning | Type / Unit |
|---|---|---|---|
| 1 | `Vehicle_ID` | Record ID | ID, no unit |
| 2 | `Car_Model` | Vehicle model | Category |
| 3 | `Battery_Type` | Battery type | Category |
| 4 | `Battery_Capacity_kWh` | Battery capacity | kWh |
| 5 | `Vehicle_Age_Months` | Vehicle age | months |
| 6 | `Total_Charging_Cycles` | Number of charging cycles | cycles |
| 7 | `Avg_Temperature_C` | Average temperature | °C |
| 8 | `Fast_Charge_Ratio` | Share of fast charging | 0 to 1, unitless |
| 9 | `Avg_Discharge_Rate_C` | Average discharge rate | C-rate |
| 10 | `Driving_Style` | Driving style | Category |
| 11 | `Internal_Resistance_Ohm` | Internal resistance | ohms (Ω) |
| 12 | `SoH_Percent` | State of Health | % |
| 13 | `Battery_Status` | Status from the dataset | Category |

The dataset has **no voltage (V), current (A), or ampere-hour (Ah) columns**. They are not used and not inferred.

**Class balance (recorded) [VERIFY against the CSV]:** 9,996 `Healthy` and 4 `Replace Required`. The program maps Healthy to class 1 and Replace Required to class 0, and shows class 0 as **"Not Healthy"**. So "Not Healthy" in the program output means the dataset label `Replace Required`.

## 6. Libraries and Their Roles

| Library | Recorded version | Role |
|---|---|---|
| Python | 3.12.10 | Programming language |
| pandas | 2.2.3 | Read and prepare table data |
| NumPy | 1.26.4 | Numeric calculations |
| Matplotlib | 3.11.2 | Create and save charts |
| Seaborn | 0.13.2 | Statistical charts and heatmaps |
| scikit-learn | 1.9.1 | Models, splitting, scaling, metrics |
| Sweetviz | 2.3.1 | Separate automatic EDA HTML report |
| setuptools | 80.9.0 | Compatibility dependency for Sweetviz |

Exact versions are in `requirements.txt`. Ridge Regression is **not** claimed here. Only mention it if you find it in the code.

## 7. Complete Workflow

The README describes `project.py` as one script that runs top to bottom, with no custom functions or classes. **[VERIFY]**

1. **Problem definition:** prints goals and the learning-only warning.
2. **Data collection:** loads the CSV into `df`; prints shape, columns, sample rows, summaries, status counts.
3. **Data preparation:** checks required fields, duplicates, and IQR outliers.
4. **EDA:** SoH summaries, Sweetviz report, charts.
5. **Feature selection:** five inputs and two targets.
6. **Train/test split:** 80% train, 20% test.
7. **Model training:** `model1`, `model2`, `model3`.
8. **Model evaluation:** metrics, train/test scores, five-fold cross-validation.
9. **Held-out prediction:** compare predicted and actual SoH for one test row.
10. **User-input prediction:** ask for five values and print results.
11. **Monitoring:** prints reminders only. No live monitoring exists.

## 8. Data Inspection, Cleaning, and Outlier Handling

- **Inspection:** shape, column names, sample rows, summary statistics, status counts.
- **Missing values and duplicates:** the recorded workflow checks required columns and removes duplicate rows and incomplete required rows. **[VERIFY exact behavior in code.]**
- **Outliers:** counted using the **IQR rule** (values far below Q1 or above Q3). The outliers are **counted but kept**, not removed.

**IQR in simple words:** Sort the values. Q1 is the 25% mark, Q3 is the 75% mark, and IQR = Q3 − Q1. A value is flagged if it is much lower than Q1 or much higher than Q3.

## 9. EDA Charts and Generated Reports

- Charts are saved as PNG files in `Reports/` and shown one at a time. **Close each chart window to continue.**
- **Sweetviz** creates `Reports/Sweetviz_EDA_Report.html`, an automatic exploratory report. Sweetviz needs a numeric or Boolean target, so a temporary True/False `Healthy` column may be made on a **copy** of the data. It does not change the CSV or the model inputs. **[VERIFY]**
- Other saved images named in the README: `deployment_prediction.png`, `model_results.png`, `confusion_matrix.png`.

## 10. Features, Targets, and Target Leakage

**Five input features (`feature_columns`):**
`Vehicle_Age_Months`, `Total_Charging_Cycles`, `Avg_Temperature_C`, `Fast_Charge_Ratio`, `Avg_Discharge_Rate_C`.

| Task | Model | Inputs | Target |
|---|---|---|---|
| Simple regression | Linear Regression | `Total_Charging_Cycles` | `SoH_Percent` |
| Multiple regression | Linear Regression | the five features | `SoH_Percent` |
| Classification | Logistic Regression | the five features | `Battery_Status` (1 = Healthy, 0 = Not Healthy) |

**Target leakage** means giving the model information that reveals the answer. `SoH_Percent` is **not** a classification input, because it would make the status too easy to guess and the score would be unrealistic.

## 11. Train/Test Split

- 80% of rows are used to **train**, 20% to **test**.
- The test set is "unseen" data used only for the final check.
- The classification split is **stratified** (keeps the Healthy / Not Healthy proportions similar). **[VERIFY]**

## 12. Models Used

**Simple Linear Regression (`model1`)** draws the best straight line from charging cycles to SoH. Roughly: `SoH ≈ a + b × cycles`.

**Multiple Linear Regression (`model2`)** extends this to five inputs: `SoH ≈ a + b1×age + b2×cycles + b3×temperature + b4×fast_ratio + b5×discharge_rate`. It uses more information, so it can fit better.

**Logistic Regression (`model3`)** predicts a **category**. It calculates a probability (0 to 1) that a battery is Healthy and picks the class. Inputs are **scaled** first so features with big numbers (like cycles) do not dominate small ones (like a 0–1 ratio).

## 13. Training and Evaluation Metrics

**Regression metrics**

| Metric | Simple meaning |
|---|---|
| MAE | Average size of the error, in SoH percentage points |
| MSE | Average of squared errors (punishes big errors) |
| RMSE | Square root of MSE; back in % units |
| R² | How much of SoH's variation the model explains (closer to 1 is better) |
| RSS | Total of all squared errors |

**Classification metrics (positive class = Not Healthy)**

| Metric | Simple meaning |
|---|---|
| Accuracy | Share of all predictions that were correct |
| Precision | Of those predicted Not Healthy, how many really were |
| Recall | Of the real Not Healthy rows, how many were found |
| F1 | Balance of precision and recall |
| Confusion matrix | Table of actual vs predicted counts |

### Previously recorded results (not recomputed in this task)

These come from one recorded run on one test split. Results can change if the data or code changes.

| Regression model | MAE | MSE | RMSE | Test R² | RSS |
|---|---:|---:|---:|---:|---:|
| Simple Linear Regression | 1.1219 | 2.6577 | 1.6302 | 0.7500 | 5315.3974 |
| Multiple Linear Regression | 1.0683 | 2.2113 | 1.4870 | 0.7920 | 4422.5896 |

Recorded best regression model: **Multiple Linear Regression**.

| Classification (Not Healthy) | Recorded value |
|---|---:|
| Accuracy | 0.9965 |
| Precision | 0.1250 |
| Recall | 1.0000 |
| F1 | 0.2222 |

| Actual \ Predicted | Not Healthy | Healthy |
|---|---:|---:|
| Not Healthy | 1 | 0 |
| Healthy | 7 | 1992 |

| Model | Train score | Test score |
|---|---:|---:|
| Simple Linear Regression (R²) | 0.7656 | 0.7500 |
| Multiple Linear Regression (R²) | 0.8009 | 0.7920 |
| Logistic Regression (accuracy) | 0.9951 | 0.9965 |

## 14. Cross-Validation and Class-Imbalance Warning

**Five-fold cross-validation** splits the data into five parts. The model trains on four and tests on the fifth, repeating five times. This shows whether the score is stable.

| Model and score | F1 | F2 | F3 | F4 | F5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Simple Linear (R²) | 0.7500 | 0.7631 | 0.7599 | 0.7678 | 0.7713 | 0.7624 |
| Multiple Linear (R²) | 0.7920 | 0.7960 | 0.7959 | 0.8065 | 0.8036 | 0.7988 |
| Logistic (accuracy) | 0.9945 | 0.9975 | 0.9970 | 0.9965 | 0.9950 | 0.9961 |

**Class-imbalance warning:** Only 4 of 10,000 rows are Not Healthy, and only 1 is in the recorded test set. A model that always says "Healthy" would score about 99.96% accuracy, so **high accuracy does not prove the rare class is handled well**. The recorded precision of 0.125 shows many false alarms (7 Healthy rows predicted Not Healthy). Four rare rows also cannot appear in every cross-validation test group.

## 15. Held-Out Prediction and User-Input Prediction

- **Held-out prediction:** the model predicts SoH for one test row, and the program compares it with that row's known SoH.
- **User-input prediction:** the console asks for five values, builds a one-row table (`new_battery`), and prints the predicted status, SoH, and Healthy probability. **[VERIFY]**

## 16. Each User Prompt Explained

| Prompt | Meaning | Unit | Example |
|---|---|---|---|
| Vehicle age | How old the vehicle is | months | 48 |
| Total charging cycles | Number of charge cycles so far | cycles | 400 |
| Average temperature | Typical operating temperature | °C | 25 |
| Fast-charge ratio | Share of fast charging | 0 to 1 | 0.49 |
| Average discharge rate | Typical discharge speed | C-rate | 1.49 |

- **Observed data range in the CSV:** **[VERIFY by checking min and max of each column]**. No ranges are stated here because they were not verified.
- **Does the code enforce these ranges?** **[VERIFY]**. The recorded context does not confirm that input validation exists. If the code only converts text to numbers, say clearly that it does **not** validate ranges.

## 17. How to Install and Run (Windows)

```powershell
git clone https://github.com/hitgalani/EV_Battery_Health_Prediction.git
cd EV_Battery_Health_Prediction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run from the **project root** (the folder with `Dataset`, `Python_Code`, `Reports`), because the script uses project-relative paths:

```powershell
python -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

Without activating the environment:

```powershell
.\.venv\Scripts\python.exe -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

If PowerShell blocks activation, use the second command. If your prompt ends with `\Python_Code>`, run `cd ..` first. **[VERIFY these commands against `README.md`.]**

## 18. Expected Output Format

When prompted, enter five numbers. The output format is:

```text
Battery status: <Healthy or Not Healthy>
Predicted State of Health: <number> %
Probability of Healthy: <number> %
Advice: <example advice text>
Learning project on synthetic data; not for real safety decisions.
```

**Previously recorded example (synthetic data, not a new run):** inputs 48, 400, 25, 0.49, 1.49 gave *Healthy*, SoH *96.34 %*, probability *100.00 %*, advice *keep using*. This is only a demonstration of the format, not real battery advice.

## 19. Limitations, Synthetic-Data Warning, and Future Improvements

**Limitations**

- Synthetic data, not verified real-world measurements.
- The course asks for real-world data, so teacher approval is needed.
- Only four `Replace Required` rows, so rare-class results are uncertain.
- Accuracy hides rare-class errors.
- Results do not prove performance on real batteries.
- This is a student demo, not safety advice.

**Future improvements**

- Use a licensed, teacher-approved dataset with enough examples of each status.
- Test on a separate, representative dataset.
- Explore remaining useful life prediction if suitable target data exists.
- Consider real-time monitoring only with validated sensors and safety review.
- Build a simple web app only after input validation and model validation.

## 21. Final Summary

This project builds two beginner models on a synthetic EV battery dataset: Linear Regression for SoH and Logistic Regression for the status label. In the recorded run, Multiple Linear Regression performed best for SoH (test R² 0.7920, MAE 1.0683 percentage points). The classification model shows 99.65% accuracy, but this is misleading because only four rows are Not Healthy. The project is for learning only and must not be used for real battery safety decisions.

---

### Checks and open items

- **Not inspected:** `project.py`, `main.py`, `main2.py`, the CSV, `requirements.txt`.
- **Not re-run:** all numbers are previously recorded values.
- **To verify before presenting:** every **[VERIFY]** item above, the purpose of `main.py` / `main2.py` / `Main/`, the data ranges, and whether input validation exists.