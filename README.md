<div align="center">

# 🔋 EV Battery Health Prediction

### Predicting Electric Vehicle battery State of Health (SoH %) with Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed%20%7C%20Updates%20Coming-brightgreen)
![Data](https://img.shields.io/badge/Data-Synthetic-yellow)
![Level](https://img.shields.io/badge/Level-Beginner%20Friendly-blueviolet)

**Hit Galani** ([@hitgalani](https://github.com/hitgalani)) &nbsp;·&nbsp; **Goti Nisarg** ([@GotiNIsarg](https://github.com/GotiNIsarg))

*SYBCA (AI) · Semester 3 · SBCCAS, Amroli*

⭐ **If this project helps you, please give it a star!** ⭐

</div>

---

> ⚠️ **Learning project only.**
> The dataset is **synthetic** (computer-generated), not measured from real batteries. This is not a certified battery diagnostic tool. **Never use it for decisions about a real car or battery.**

---

## 📑 Table of Contents

1. [About the Project](#-about-the-project)
2. [College and Course](#-college-and-course)
3. [Project Status and Timeline](#-project-status-and-timeline)
4. [Screenshots](#️-screenshots)
5. [Dataset](#-dataset)
6. [Machine Learning Approach](#-machine-learning-approach)
7. [Workflow](#-workflow)
8. [Results](#-results)
9. [Class Imbalance Warning](#️-class-imbalance-warning)
10. [Project Files and Code Guide](#-project-files-and-code-guide)
11. [Installation and Usage](#-installation-and-usage)
12. [Technologies](#-technologies)
13. [Learning Resources](#-learning-resources)
14. [How I Used AI to Learn](#-how-i-used-ai-to-learn)
15. [Future Updates](#-future-updates)
16. [Support This Project](#-support-this-project)
17. [Glossary](#-glossary)
18. [Acknowledgements](#-acknowledgements)

---

## 🚀 About the Project

The **State of Health (SoH)** tells us how much life an electric-car battery has left compared with a new one: 100% is like new, and lower values mean more wear.

This project uses **five simple measurements** to:

1. **Predict a number:** the battery's SoH (%) using **Linear Regression**.
2. **Predict a label:** the battery's status (*Healthy* or *Not Healthy*) using **Logistic Regression**.

It follows the full beginner ML journey: **load → clean → explore → split → train → test → predict.**

---

## 🎓 College and Course

| Detail | Info |
|---|---|
| **College** | Sutex Bank College of Computer Applications & Science (SBCCAS), Amroli, Surat, Gujarat |
| **University** | Veer Narmad South Gujarat University (VNSGU) |
| **Course** | BCA (Artificial Intelligence), **SYBCA** |
| **Semester** | **Semester 3** |
| **Project type** | Academic team project |

### 👥 Team

| Name | Roll No. | GitHub |
|---|---:|---|
| **Hit Galani** (team leader) | 577 | [@hitgalani](https://github.com/hitgalani) |
| **Goti Nisarg** | 574 | [@GotiNIsarg](https://github.com/GotiNIsarg) |

We are SYBCA AI students, and this project is part of our Semester 3 learning in Python, Machine Learning and data science.

---

## 📅 Project Status and Timeline

**Current status: ✅ Completed (version 1).** The full pipeline works, from data loading to user prediction. **More updates are coming**, see [Future Updates](#-future-updates).

| Phase | What we did | Status |
|---|---|---|
| 1. Learn | Studied supervised ML, regression and classification | ✅ Done |
| 2. Data | Chose and inspected the Kaggle dataset | ✅ Done |
| 3. Explore | Cleaning, outlier check, EDA charts, Sweetviz report | ✅ Done |
| 4. Build | Simple, Multiple and Logistic models | ✅ Done |
| 5. Evaluate | Test set, metrics, five-fold cross-validation | ✅ Done |
| 6. Document | README, report and learning website | ✅ Done |
| 7. Improve | Input checks, better data, more charts | 🔜 Coming |

---

## 🖼️ Screenshots

Open the full visual reports in the repo:

- 📊 [Project report (HTML)](Reports/EV_Battery_Project_Report.html)
- 🍬 [Sweetviz EDA report (HTML)](Reports/Sweetviz_EDA_Report.html)

<!--
SCREENSHOT TEMPLATE: replace the file names below with your real images in Reports/,
then delete the comment markers (the first and last line of this block).

<div align="center">

<img src="Reports/soh_vs_cycles.png" alt="Scatter chart of SoH against charging cycles" width="720">

*Figure 1: SoH (%) against total charging cycles*

<img src="Reports/correlation_heatmap.png" alt="Correlation heatmap of the battery features" width="720">

*Figure 2: Correlation between the battery features*

</div>
-->

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

## 🧠 Machine Learning Approach

| Task | Model | Inputs | Target |
|---|---|---|---|
| Simple regression | Linear Regression | `Total_Charging_Cycles` | `SoH_Percent` |
| Multiple regression | Linear Regression | 5 features | `SoH_Percent` |
| Classification | Logistic Regression | 5 features | `Battery_Status` |

**The 5 input features:** `Vehicle_Age_Months`, `Total_Charging_Cycles`, `Avg_Temperature_C`, `Fast_Charge_Ratio`, `Avg_Discharge_Rate_C`

- **Split:** 80% training, 20% testing.
- **Scaling:** inputs are scaled for Logistic Regression so large numbers (cycles) do not dominate small ones (a 0 to 1 ratio).
- **No target leakage:** `SoH_Percent` is *not* used to predict status, because it would reveal the answer.

---

## 🔄 Workflow

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

## 📈 Results

> These are from **one recorded run on one test split**. They may change if the data or code changes.

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

---

## ⚖️ Class Imbalance Warning

The data is **very unbalanced**: **9,996** `Healthy` rows and only **4** `Replace Required` rows.

A model that always says "Healthy" would score about **99.96% accuracy**, so our 99.65% accuracy does **not** prove the rare class is handled well. The low precision (0.125) means 7 Healthy batteries were wrongly flagged.

💡 **Lesson:** accuracy alone can fool you. Always check precision, recall and the confusion matrix.

---

## 📂 Project Files and Code Guide

```text
EV_Battery_Health_Prediction/
├── Dataset/
│   └── EV_Battery_Dataset.csv
├── Python_Code/
│   └── project.py
├── Reports/
│   ├── EV_Battery_Project_Report.html
│   ├── Sweetviz_EDA_Report.html
│   └── *.png
├── Learning_Website/
│   ├── index.html
│   └── images/
├── README.md
├── requirements.txt
└── .gitignore
```

| File / folder | What it is |
|---|---|
| `Dataset/EV_Battery_Dataset.csv` | The data used for training and testing |
| `Python_Code/project.py` | The main script. It does everything in the [Workflow](#-workflow), from loading to prediction |
| `Reports/` | Saved charts (`.png`), the project report and the Sweetviz EDA report |
| `Learning_Website/` | An offline beginner guide with images |
| `requirements.txt` | Exact library versions |
| `.gitignore` | Files Git should ignore, such as `.venv/` |

**About the models:** the three models (Simple Linear, Multiple Linear, Logistic) are trained inside `project.py` each time it runs. This README does not document a separately saved model file.

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

Resources that help you understand each part of this project:

| Topic | Resource |
|---|---|
| Machine Learning basics, Python, pandas | [Kaggle Learn (free courses)](https://www.kaggle.com/learn) |
| Linear and Logistic Regression, metrics | [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) |
| Data tables | [pandas documentation](https://pandas.pydata.org/docs/) |
| Number calculations | [NumPy documentation](https://numpy.org/doc/) |
| Charts | [Matplotlib](https://matplotlib.org/stable/) · [Seaborn](https://seaborn.pydata.org/) |
| Our dataset | [Kaggle dataset page](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) |

### 🎥 Videos we learned from

> Add the video titles and links you really watched here.

| Topic | Video title | Link |
|---|---|---|
| Linear Regression | *(add title)* | *(add link)* |
| Logistic Regression | *(add title)* | *(add link)* |
| Train/test split and cross-validation | *(add title)* | *(add link)* |

---

## 🤖 How I Used AI to Learn

I used AI assistants as a **study partner**, not as a replacement for learning. They helped me:

- 🧠 Understand each step of the ML workflow in simple words.
- 💻 Read and understand the code in `project.py`, section by section.
- 📊 Understand what MAE, RMSE, R², precision, recall and F1 mean.
- 📝 Improve the writing and layout of this README.

I ran the code myself, and every result in this README comes from our own run. Using AI to learn is fine, but always **understand what you submit and check it yourself**.

---

## 🔜 Future Updates

The project is complete, and **more updates are coming**:

- [ ] Friendly input checks (for example, reject a fast-charge ratio above 1)
- [ ] Clearer chart titles, labels and screenshots in this README
- [ ] Better Learning Website layout
- [ ] A licensed, teacher-approved dataset with enough examples of every status
- [ ] Careful handling of class imbalance, explained simply
- [ ] Remaining useful life prediction, if suitable data exists

⭐ **Star** or **watch** this repository to see updates as they arrive.

---

## ⭐ Support This Project

If this project helped you learn, please **give it a star** (the ⭐ button at the top right of the page).

**How a star helps**
- 🌱 It tells us the project is useful and keeps us motivated to add updates.
- 🔍 It helps other students find a beginner-friendly ML project.
- 📌 It bookmarks the repo on your GitHub, so you can come back to it later.

**How it helps you**
- You get a simple project to learn from and to build on.
- Your starred list becomes your personal library of projects.
- Starring or watching shows you our future updates.

You can also **fork** the project to practise, and open an **Issue** with suggestions or questions. 🙏

---

## 📖 Glossary

| Word | Easy meaning |
|---|---|
| **EV** | Electric vehicle |
| **SoH** | State of Health: how healthy a battery is compared with new |
| **Synthetic data** | Data made by a computer, not measured from real things |
| **Feature** | An input the model uses |
| **Target** | The answer the model predicts |
| **Regression** | Predicting a number |
| **Classification** | Predicting a category |
| **Cross-validation** | Testing on several splits to check that scores are stable |
| **Target leakage** | Giving the model a clue that reveals the answer |
| **EDA** | Exploratory Data Analysis: looking at data before modelling |
| **Outlier** | A value far from the rest |
| **Class imbalance** | One category is far more common than the other |

---

## 🙏 Acknowledgements

- **SBCCAS, Amroli** and our teachers for guidance and support
- [Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) for the dataset
- The open-source Python and ML community

> 📜 **License:** not yet chosen. Add a `LICENSE` file and check the Kaggle dataset's license before sharing widely.

---

<div align="center">

**Developed by Hit Galani and Goti Nisarg** · SYBCA (AI), Semester 3, SBCCAS Amroli

*Learning project only. Not for real battery safety decisions.*

⭐ **Thank you for visiting! If you liked it, please star the repo.** ⭐

</div>
