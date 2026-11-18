<div align="center">

# 🔋 EV Battery Health Prediction

### Predicting Electric Vehicle battery State of Health (SoH %) with Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?logo=pandas&logoColor=white)
![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)
![Semester 3](https://img.shields.io/badge/Semester%203-Supervised%20Learning%20Complete-brightgreen)
![Semester 4](https://img.shields.io/badge/Semester%204-Planned-lightgrey)
![Data](https://img.shields.io/badge/Data-Synthetic-yellow)

**Hit Galani** ([@hitgalani](https://github.com/hitgalani)) &nbsp;·&nbsp; **Goti Nisarg** ([@GotiNIsarg](https://github.com/GotiNIsarg))

*SYBCA (AI) · SBCCAS, Amroli · Surat, Gujarat*

</div>

---

> ⚠️ **Learning project only.**
> The dataset is **synthetic** (computer-generated), not measured from real batteries. This is not a battery diagnostic tool. **Never use it for decisions about a real car or battery.**

---

## 📑 Contents

1. [About](#-about)
2. [Project Roadmap](#-project-roadmap)
3. [Results (Version 1)](#-results-version-1)
4. [Dataset](#-dataset)
5. [How It Works](#-how-it-works)
6. [Project Structure](#-project-structure)
7. [Installation and Usage](#-installation-and-usage)
8. [Technologies](#-technologies)
9. [Learning Resources](#-learning-resources)
10. [Changelog](#-changelog)
11. [Team](#-team)
12. [Glossary](#-glossary)

---

## 🚀 About

**State of Health (SoH)** shows how much life an electric-car battery has left compared with a new one. 100% is like new; lower values mean more wear.

This project uses **five simple measurements** to:

1. **Predict a number**: the battery's SoH (%) using **Linear Regression**.
2. **Predict a label**: the battery's status (*Healthy* or *Not Healthy*) using **Logistic Regression**.

It follows the full beginner ML journey: **load → clean → explore → split → train → test → predict.**

---

## 🗺️ Project Roadmap

This is a **long-term learning project**. It grows as we learn, one semester at a time.

| Semester | Focus | Status |
|---|---|---|
| **Semester 3** | **Supervised learning**: Simple, Multiple and Logistic Regression, evaluation, cross-validation, EDA report | ✅ **Complete (Version 1)** |
| **Semester 4** | **Unsupervised learning** and **deployment**, applied to this same project | 🔜 Planned |
| **After Semester 4** | Update the code and the dataset using what we learned | 🔜 Planned |

### Planned for Semester 4
- Learn unsupervised learning and add it to this project
- Learn deployment and make the model usable outside the terminal
- Add friendly input checks (for example, reject a fast-charge ratio above 1)
- Replace the dataset with a better one that has enough examples of every status
- Handle class imbalance properly
- Add screenshots and clearer charts to this README

> The Semester 4 plan may change as we learn. This README is updated whenever the project changes (see [Changelog](#-changelog)).

---

## 📈 Results (Version 1)

> These come from **one recorded run on one test split**. They may change if the data or code changes.

### Regression (predicting SoH %)

| Model | MAE | MSE | RMSE | Test R² |
|---|---:|---:|---:|---:|
| Simple Linear Regression | 1.1219 | 2.6577 | 1.6302 | 0.7500 |
| **Multiple Linear Regression** 🏆 | 1.0683 | 2.2113 | 1.4870 | 0.7920 |

### Classification (positive class = Not Healthy)

| Metric | Value |
|---|---:|
| Accuracy | 99.65% |
| Precision | 12.50% |
| Recall | 100% |
| F1 score | 22.22% |

| Actual ↓ / Predicted → | Not Healthy | Healthy |
|---|---:|---:|
| **Not Healthy** | 1 | 0 |
| **Healthy** | 7 | 1992 |

### Five-Fold Cross-Validation

| Model | F1 | F2 | F3 | F4 | F5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Simple Linear (R²) | 0.7500 | 0.7631 | 0.7599 | 0.7678 | 0.7713 | 0.7624 |
| Multiple Linear (R²) | 0.7920 | 0.7960 | 0.7959 | 0.8065 | 0.8036 | 0.7988 |
| Logistic (accuracy) | 0.9945 | 0.9975 | 0.9970 | 0.9965 | 0.9950 | 0.9961 |

### ⚖️ Class imbalance warning

The data is **very unbalanced**: **9,996** `Healthy` rows and only **4** `Replace Required` rows. A model that always says "Healthy" would score about **99.96% accuracy**, so our 99.65% does **not** prove the rare class is handled well. The low precision (12.5%) means 7 Healthy batteries were wrongly flagged.

💡 **Lesson:** accuracy alone can fool you. Always check precision, recall and the confusion matrix. Fixing this is a goal for the next version.

### What the metrics mean

| Metric | Simple meaning |
|---|---|
| **MAE** | Average error size, in SoH percentage points |
| **MSE / RMSE** | Squared error and its square root (big mistakes count more) |
| **R²** | How much of SoH's variation the model explains (closer to 1 is better) |
| **Accuracy** | Share of all predictions that were right |
| **Precision** | Of those predicted Not Healthy, how many really were |
| **Recall** | Of the real Not Healthy rows, how many were found |
| **F1** | A balance of precision and recall |

### Visual reports

- 📊 [Project report (HTML)](Reports/EV_Battery_Project_Report.html)
- 🍬 [Sweetviz EDA report (HTML)](Reports/Sweetviz_EDA_Report.html)

---

## 📊 Dataset

| Detail | Value |
|---|---|
| Name | Electric Vehicle (EV) Battery Degradation and Charge |
| Source | [Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) |
| File | `Dataset/EV_Battery_Dataset.csv` |
| Size | 10,000 rows × 13 columns |
| Type | **Synthetic** |

| # | Column | Meaning | Unit |
|---|---|---|---|
| 1 | `Vehicle_ID` | Record ID | none |
| 2 | `Car_Model` | Vehicle model | category |
| 3 | `Battery_Type` | Battery type | category |
| 4 | `Battery_Capacity_kWh` | Battery capacity | kWh |
| 5 | `Vehicle_Age_Months` | Vehicle age | months |
| 6 | `Total_Charging_Cycles` | Charging cycles so far | cycles |
| 7 | `Avg_Temperature_C` | Average temperature | °C |
| 8 | `Fast_Charge_Ratio` | Share of fast charging | 0 to 1 |
| 9 | `Avg_Discharge_Rate_C` | Average discharge rate | C-rate |
| 10 | `Driving_Style` | Driving style | category |
| 11 | `Internal_Resistance_Ohm` | Internal resistance | Ω |
| 12 | `SoH_Percent` | State of Health | % |
| 13 | `Battery_Status` | Health status from the dataset | category |

**Labels:** `Healthy` = class 1. `Replace Required` = class 0, shown in the program as **Not Healthy**.

---

## 🧠 How It Works

| Task | Model | Inputs | Target |
|---|---|---|---|
| Simple regression | Linear Regression | `Total_Charging_Cycles` | `SoH_Percent` |
| Multiple regression | Linear Regression | 5 features | `SoH_Percent` |
| Classification | Logistic Regression | 5 features | `Battery_Status` |

**The 5 input features:** `Vehicle_Age_Months`, `Total_Charging_Cycles`, `Avg_Temperature_C`, `Fast_Charge_Ratio`, `Avg_Discharge_Rate_C`

- **Split:** 80% training, 20% testing.
- **Scaling:** inputs are scaled for Logistic Regression so large numbers (cycles) do not dominate small ones (a 0 to 1 ratio).
- **No target leakage:** `SoH_Percent` is *not* used to predict status, because it would reveal the answer.

### Workflow

| Step | What happens |
|---|---|
| 1. Problem definition | Prints goals and the learning-only warning |
| 2. Data collection | Loads the CSV; shows shape, columns, samples, status counts |
| 3. Data preparation | Checks required fields, duplicates and outliers (IQR rule, counted but kept) |
| 4. EDA | SoH summaries, Sweetviz report, charts |
| 5. Feature selection | Five inputs, two targets |
| 6. Train/test split | 80% / 20% |
| 7. Model training | Simple, Multiple and Logistic models |
| 8. Evaluation | Metrics, train/test scores, five-fold cross-validation |
| 9. Held-out prediction | Predicted vs actual SoH for one test row |
| 10. User prediction | Asks for five values and prints the result |
| 11. Monitoring | Prints reminders only. **No live monitoring exists.** |

---

## 📂 Project Structure

```text
EV_Battery_Health_Prediction/
├── Dataset/
│   └── EV_Battery_Dataset.csv
├── Python_Code/
│   └── project.py            # main script (Version 1)
├── Reports/                  # charts (.png) and HTML reports
├── Report in Main Code/      # extra report outputs
├── Learning_Website/         # offline beginner guide with images
├── main.py                   # additional script
├── main2.py                  # additional script
├── requirements.txt
├── README.md
└── .gitignore
```

| File / folder | What it is |
|---|---|
| `Python_Code/project.py` | The main script. It does everything in the workflow above, from loading to prediction |
| `Reports/` | Saved charts, the project report and the Sweetviz EDA report |
| `Learning_Website/` | An offline beginner guide with images |
| `requirements.txt` | Exact library versions |

Models are trained inside the script every time it runs. No separate saved model file exists yet (deployment is planned for Semester 4).

---

## ⚙️ Installation and Usage

Instructions are for **Windows (PowerShell)**.

```powershell
# 1. Get the project
git clone https://github.com/hitgalani/EV_Battery_Health_Prediction.git
cd EV_Battery_Health_Prediction

# 2. Create and activate a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the libraries
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run the project (from the project root folder)
python Python_Code\project.py
```

**Tips**
- If PowerShell blocks activation, run `.\.venv\Scripts\python.exe Python_Code\project.py` directly.
- Run from the **project root**, because the script uses project-relative paths.
- Charts open one at a time. **Close each chart window to continue.**

### Prediction prompts

| Prompt | Unit | Example |
|---|---|---|
| Vehicle age | months | 48 |
| Total charging cycles | cycles | 400 |
| Average temperature | °C | 25 |
| Fast-charge ratio | 0 to 1 | 0.49 |
| Average discharge rate | C-rate | 1.49 |

The program prints the battery status, predicted SoH (%), probability of Healthy, and a short educational advice line.

---

## 🧰 Technologies

| Library | Version | Role |
|---|---|---|
| Python | 3.12.10 | Programming language |
| pandas | 2.2.3 | Data tables |
| NumPy | 1.26.4 | Number calculations |
| Matplotlib | 3.11.2 | Charts |
| Seaborn | 0.13.2 | Statistical charts |
| scikit-learn | 1.9.1 | Models, splitting, scaling, metrics |
| Sweetviz | 2.3.1 | Automatic EDA report |
| setuptools | 80.9.0 | Compatibility helper for Sweetviz |

---

## 📚 Learning Resources

| Topic | Resource |
|---|---|
| ML basics, Python, pandas | [Kaggle Learn](https://www.kaggle.com/learn) |
| Linear and Logistic Regression, metrics | [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) |
| Data tables | [pandas documentation](https://pandas.pydata.org/docs/) |
| Number calculations | [NumPy documentation](https://numpy.org/doc/) |
| Charts | [Matplotlib](https://matplotlib.org/stable/) · [Seaborn](https://seaborn.pydata.org/) |
| Dataset | [Kaggle dataset page](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) |

We also use AI assistants as a study partner to understand the workflow, code and metrics. We run the code ourselves, and every result here comes from our own run.

---

## 📝 Changelog

| Version | When | What changed |
|---|---|---|
| **v1.0.0** | Semester 3 | First complete version: supervised learning pipeline, EDA reports, learning website |
| v1.0.2 | Semester 3 | Restored project report, final README and roadmap |
| v2.0.0 | Semester 4 (planned) | Unsupervised learning, deployment, updated code and dataset |

---

## 👥 Team

| Name | Roll No. | GitHub |
|---|---:|---|
| **Hit Galani** (team leader) | 577 | [@hitgalani](https://github.com/hitgalani) |
| **Goti Nisarg** | 574 | [@GotiNIsarg](https://github.com/GotiNIsarg) |

**College:** Sutex Bank College of Computer Applications & Science (SBCCAS), Amroli, Surat, Gujarat
**University:** Veer Narmad South Gujarat University (VNSGU)
**Course:** BCA (Artificial Intelligence), SYBCA

Thanks to our teachers at SBCCAS, [Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) for the dataset, and the open-source Python and ML community.

⭐ If this project helps you learn, please star the repo. **Star or watch** it to see updates.

> 📜 **License:** not yet chosen. Add a `LICENSE` file and check the Kaggle dataset's license before sharing widely.

---

## 📖 Glossary

| Word | Easy meaning |
|---|---|
| **EV** | Electric vehicle |
| **SoH** | State of Health: how healthy a battery is compared with new |
| **Synthetic data** | Data made by a computer, not measured from real things |
| **Feature** | An input the model uses |
| **Target** | The answer the model predicts |
| **Supervised learning** | Learning from data that already has the correct answers |
| **Unsupervised learning** | Finding patterns in data that has no answers |
| **Deployment** | Making a model usable by other people |
| **Regression** | Predicting a number |
| **Classification** | Predicting a category |
| **Cross-validation** | Testing on several splits to check that scores are stable |
| **Target leakage** | Giving the model a clue that reveals the answer |
| **EDA** | Exploratory Data Analysis: looking at data before modelling |
| **Outlier** | A value far from the rest |
| **Class imbalance** | One category is far more common than the other |

---

<div align="center">

**Developed by Hit Galani and Goti Nisarg** · SYBCA (AI), SBCCAS Amroli

*Learning project only. Not for real battery safety decisions.*

</div>
