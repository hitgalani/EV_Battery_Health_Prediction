<div align="center">

# 🔋⚡ EV Battery Health Prediction 🚗💚

### A friendly beginner machine-learning project that learns how healthy an electric-car battery is

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?logo=pandas&logoColor=white)
![Type](https://img.shields.io/badge/Project-Learning%20Only-brightgreen)
![Data](https://img.shields.io/badge/Data-Synthetic-yellow)
![Team](https://img.shields.io/badge/Team-2%20People-purple)

**👑 Project Head:** [Hit Galani](https://github.com/hitgalani) &nbsp;|&nbsp; **🤝 Co-Developer:** [Goti Nisarg](https://github.com/GotiNIsarg)

🌍 *Made for everyone: students, kids, beginners, teachers, and curious people all over the world!* 🌍

</div>

---

## 🚨 Please Read First: Safety and Honesty Notice

> ⚠️ **This is a LEARNING project only.**
>
> - 🧪 The dataset is **synthetic** (made-up, computer-generated). It is **not** from real batteries.
> - 🔧 This is **not** a real battery inspection, a certified diagnosis, or a safety tool.
> - 🚫 **Never** use this project to decide anything about a real car or battery.
> - 🎓 The course asks for real-world data, so **get teacher approval** before submitting this dataset.

We are proud to be honest about what this project can and cannot do. 💙

---

## 📑 Table of Contents

1. [👋 What Is This Project?](#-what-is-this-project)
2. [👥 Meet the Team](#-meet-the-team)
3. [🎯 Project Goals](#-project-goals)
4. [🗂️ Folder Layout](#️-folder-layout)
5. [📊 The Dataset](#-the-dataset)
6. [🧰 Tools We Use](#-tools-we-use)
7. [🔄 How the Project Works](#-how-the-project-works-step-by-step)
8. [🧠 The Three Models in Simple Words](#-the-three-models-in-simple-words)
9. [📈 Results](#-results-recorded-run)
10. [⚖️ Class Imbalance Warning](#️-class-imbalance-warning)
11. [🚀 How to Install and Run (Windows)](#-how-to-install-and-run-windows)
12. [⌨️ Using the Prediction Prompts](#️-using-the-prediction-prompts)
13. [🤝 Teamwork Guide: Two People, One Project](#-teamwork-guide-two-people-one-project)
14. [🎨 Layout and Looks Guidelines](#-layout-and-looks-guidelines)
15. [💻 Code Guidelines](#-code-guidelines)
16. [🌿 Git and GitHub Workflow](#-git-and-github-workflow)
17. [🧯 Common Problems and Fixes](#-common-problems-and-fixes)
18. [🎉 Fun Corner](#-fun-corner)
19. [🗺️ Roadmap](#️-roadmap-future-ideas)
20. [📝 Changelog and Update Log](#-changelog-and-update-log)
21. [📚 Learning Log](#-learning-log)
22. [📖 Mini Glossary](#-mini-glossary)
23. [❓ FAQ](#-faq)
24. [🙏 Credits and Thanks](#-credits-and-thanks)
25. [✅ Things to Double-Check](#-things-to-double-check-before-presenting)

---

## 👋 What Is This Project?

Imagine your phone battery. When it is new, it lasts all day. After two years, it lasts less. 📱🔋

Electric-car batteries get older too. The **State of Health (SoH)** tells us how much life a battery has left compared with a brand-new one:

| SoH | What it means |
|---|---|
| 💯 100% | Like new! |
| 🙂 90% | Still great |
| 😐 80% | Getting tired |
| 😟 Lower | More wear |

This project uses **five simple measurements** to:

1. 🔢 **Predict the number:** the battery's **State of Health (%)** using **Linear Regression**.
2. 🏷️ **Predict the label:** the battery's **status** (Healthy or Not Healthy) using **Logistic Regression**.

It walks through a full beginner machine-learning journey: **load → clean → explore → split → train → test → predict.** 🚶‍♂️➡️🎓

---

## 👥 Meet the Team

| 👤 Person | 🐙 GitHub | 🎭 Role | 🛠️ Main Jobs |
|---|---|---|---|
| **Hit Galani** 👑 | [`hitgalani`](https://github.com/hitgalani) | **Project Head and Manager** | Owns the project, manages Git and GitHub, reviews and merges changes, decides the direction, keeps the main branch safe |
| **Goti Nisarg** 🤝 | [`GotiNIsarg`](https://github.com/GotiNIsarg) | **Co-Developer** | Fixes code, adds extra updates, improves project layout and looks, writes extra guides and instructions |

### 💬 Team Motto

> **"Learn together, build together, improve a little every day!"** 🌱

Both of us are **learning, improving, and updating** this project step by step. Mistakes are okay. Mistakes are how we learn! 💪

---

## 🎯 Project Goals

1. 🔍 Inspect and prepare the supplied CSV file.
2. 📊 Explore the values with tables and charts.
3. 📉 Train **simple** and **multiple** Linear Regression models for SoH.
4. 🏷️ Train **Logistic Regression** for the battery-status label.
5. 🧪 Evaluate with a held-out test set and **five-fold cross-validation**.
6. 🧑‍💻 Ask the user for five measurements and show an **educational** prediction.
7. 🌍 Make the project easy for **anyone in the world** to read, run, and learn from.

---

## 🗂️ Folder Layout

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

| 📁 Item | 🎯 Purpose |
|---|---|
| `Dataset/EV_Battery_Dataset.csv` | 📦 The data used for training and testing |
| `Python_Code/project.py` | 🐍 The main script to run |
| `Reports/` | 🖼️ Saved charts, result tables, and the Sweetviz HTML report |
| `Learning_Website/` | 🌐 Offline beginner guide with images |
| `README.md` | 📖 This file! |
| `requirements.txt` | 📌 Exact library versions |
| `.gitignore` | 🙈 Files Git should ignore (like `.venv/`) |

> 📝 **Note for the team:** In the local VS Code workspace there are also `main.py`, `main2.py`, and a `Main/` folder. Their purpose is not yet documented. See [Things to Double-Check](#-things-to-double-check-before-presenting).

---

## 📊 The Dataset

| Detail | Value |
|---|---|
| 📛 Name | Electric Vehicle (EV) Battery Degradation and Charge |
| 🔗 Source | [Kaggle dataset](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge) |
| 📄 File | `Dataset/EV_Battery_Dataset.csv` |
| 📏 Recorded size | 10,000 rows × 13 columns |
| 🧪 Data type | **Synthetic** |

### 🧾 The 13 Columns

| # | Column | Meaning | Unit |
|---|---|---|---|
| 1 | `Vehicle_ID` | Record ID | none |
| 2 | `Car_Model` | Vehicle model | category |
| 3 | `Battery_Type` | Battery type | category |
| 4 | `Battery_Capacity_kWh` | Battery capacity | kWh |
| 5 | `Vehicle_Age_Months` | Vehicle age | months |
| 6 | `Total_Charging_Cycles` | Number of charging cycles | cycles |
| 7 | `Avg_Temperature_C` | Average temperature | °C |
| 8 | `Fast_Charge_Ratio` | Share of fast charging | 0 to 1 |
| 9 | `Avg_Discharge_Rate_C` | Average discharge rate | C-rate |
| 10 | `Driving_Style` | Driving style | category |
| 11 | `Internal_Resistance_Ohm` | Internal resistance | ohms (Ω) |
| 12 | `SoH_Percent` | State of Health | % |
| 13 | `Battery_Status` | Status from the dataset | category |

🔌 **No voltage (V), current (A), or ampere-hour (Ah) columns exist.** We never use or invent them.

### 🏷️ Label Meaning

- ✅ **Healthy** = class `1`
- ❌ **Replace Required** = class `0`, shown in the program as **"Not Healthy"**

---

## 🧰 Tools We Use

| 🧰 Library | 📌 Version | 🎯 Role |
|---|---|---|
| 🐍 Python | 3.12.10 | The programming language |
| 🐼 pandas | 2.2.3 | Read and prepare table data |
| 🔢 NumPy | 1.26.4 | Number calculations |
| 📈 Matplotlib | 3.11.2 | Create and save charts |
| 🎨 Seaborn | 0.13.2 | Pretty statistical charts and heatmaps |
| 🤖 scikit-learn | 1.9.1 | Models, splitting, scaling, metrics |
| 🍬 Sweetviz | 2.3.1 | Automatic EDA HTML report |
| 🧩 setuptools | 80.9.0 | Compatibility helper for Sweetviz |

Exact versions live in `requirements.txt`.

---

## 🔄 How the Project Works (Step by Step)

```text
 1️⃣ Define problem  →  2️⃣ Load data  →  3️⃣ Clean data  →  4️⃣ Explore (EDA)
                                                                │
 8️⃣ Evaluate  ←  7️⃣ Train models  ←  6️⃣ Split 80/20  ←  5️⃣ Pick features
      │
 9️⃣ Predict one test row  →  🔟 Predict from YOUR input  →  1️⃣1️⃣ Monitoring reminders
```

| Step | What happens |
|---|---|
| 1️⃣ Problem definition | Prints goals and the learning-only warning |
| 2️⃣ Data collection | Loads the CSV; shows shape, columns, sample rows, summaries, status counts |
| 3️⃣ Data preparation | Checks required fields, duplicates, and outliers (IQR rule) |
| 4️⃣ EDA | SoH summaries, Sweetviz report, charts |
| 5️⃣ Feature selection | Five inputs and two targets |
| 6️⃣ Train/test split | 80% train, 20% test |
| 7️⃣ Model training | Three models: `model1`, `model2`, `model3` |
| 8️⃣ Model evaluation | Metrics, train/test scores, five-fold cross-validation |
| 9️⃣ Held-out prediction | Compare predicted and actual SoH for one test row |
| 🔟 User-input prediction | Ask for five values and print results |
| 1️⃣1️⃣ Monitoring | Prints reminders only. **No live monitoring exists.** |

### 🔍 About Outliers (IQR in Simple Words)

Sort the values. **Q1** is the 25% mark, **Q3** is the 75% mark, and **IQR = Q3 − Q1**. A value is flagged if it is far below Q1 or far above Q3. Outliers are **counted but kept**, not removed. 🧹➡️🙅

### 🍬 About Sweetviz

It builds `Reports/Sweetviz_EDA_Report.html`, an automatic explore-the-data report. A temporary True/False `Healthy` column may be made on a **copy** of the data, so the original CSV and model inputs stay untouched. 🔒

---

## 🧠 The Three Models in Simple Words

### 📏 Model 1: Simple Linear Regression
Draws the best straight line from **charging cycles** to **SoH**.
`SoH ≈ a + b × cycles`

### 📐 Model 2: Multiple Linear Regression
Same idea, but uses **all five inputs** for a smarter guess.
`SoH ≈ a + b1×age + b2×cycles + b3×temperature + b4×fast_ratio + b5×discharge_rate`

### 🏷️ Model 3: Logistic Regression
Predicts a **category** (Healthy or Not Healthy). It works out a probability from 0 to 1, then picks a class. Inputs are **scaled** first so big numbers (like cycles) do not bully small ones (like a 0–1 ratio). 🥊➡️🤝

### 🧩 Inputs and Targets

| 🎯 Task | 🤖 Model | 📥 Inputs | 🏁 Target |
|---|---|---|---|
| Simple regression | Linear Regression | `Total_Charging_Cycles` | `SoH_Percent` |
| Multiple regression | Linear Regression | 5 features | `SoH_Percent` |
| Classification | Logistic Regression | 5 features | `Battery_Status` |

**The 5 input features:**
`Vehicle_Age_Months`, `Total_Charging_Cycles`, `Avg_Temperature_C`, `Fast_Charge_Ratio`, `Avg_Discharge_Rate_C`

### 🕵️ What Is Target Leakage?
Giving the model a clue that reveals the answer. `SoH_Percent` is **not** used to predict status, because that would make the test too easy and the score unrealistic. 🚫🔑

---

## 📈 Results (Recorded Run)

> 📌 These numbers are from **one recorded run on one test split**. They can change if the data or code changes. They are shown here for learning.

### 📉 Regression (predicting SoH %)

| Model | MAE | MSE | RMSE | Test R² | RSS |
|---|---:|---:|---:|---:|---:|
| Simple Linear Regression | 1.1219 | 2.6577 | 1.6302 | 0.7500 | 5315.3974 |
| Multiple Linear Regression 🏆 | 1.0683 | 2.2113 | 1.4870 | 0.7920 | 4422.5896 |

🏆 **Best regression model:** Multiple Linear Regression.

### 🏷️ Classification (positive class = Not Healthy)

| Metric | Value |
|---|---:|
| Accuracy | 0.9965 |
| Precision | 0.1250 |
| Recall | 1.0000 |
| F1 | 0.2222 |

| Actual ↓ / Predicted → | Not Healthy | Healthy |
|---|---:|---:|
| **Not Healthy** | 1 | 0 |
| **Healthy** | 7 | 1992 |

### 🏋️ Train vs Test Scores

| Model | Train | Test |
|---|---:|---:|
| Simple Linear (R²) | 0.7656 | 0.7500 |
| Multiple Linear (R²) | 0.8009 | 0.7920 |
| Logistic (accuracy) | 0.9951 | 0.9965 |

### 🔁 Five-Fold Cross-Validation

| Model | F1 | F2 | F3 | F4 | F5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Simple Linear (R²) | 0.7500 | 0.7631 | 0.7599 | 0.7678 | 0.7713 | 0.7624 |
| Multiple Linear (R²) | 0.7920 | 0.7960 | 0.7959 | 0.8065 | 0.8036 | 0.7988 |
| Logistic (accuracy) | 0.9945 | 0.9975 | 0.9970 | 0.9965 | 0.9950 | 0.9961 |

### 📖 What the Metrics Mean

| Metric | Simple meaning |
|---|---|
| **MAE** | Average error size, in SoH percentage points |
| **MSE** | Average of squared errors (big mistakes hurt more) |
| **RMSE** | Square root of MSE, back in % units |
| **R²** | How much of SoH's ups and downs the model explains (closer to 1 = better) |
| **RSS** | Total of all squared errors |
| **Accuracy** | Share of all predictions that were right |
| **Precision** | Of those predicted Not Healthy, how many really were |
| **Recall** | Of the real Not Healthy rows, how many we found |
| **F1** | A balance of precision and recall |
| **Confusion matrix** | Table of actual vs predicted counts |

---

## ⚖️ Class Imbalance Warning

🐘🐁 Our data is **very unbalanced**:

- ✅ **9,996** rows are `Healthy`
- ❌ **4** rows are `Replace Required`

A lazy model that **always says "Healthy"** would score about **99.96% accuracy**. So the high accuracy (99.65%) **does not prove** the rare class is handled well.

The low precision of **0.125** shows many false alarms (7 Healthy batteries called Not Healthy). Also, 4 rare rows cannot appear in every cross-validation group.

💡 **Lesson learned:** *Accuracy alone can fool you!* Always look at precision, recall, and the confusion matrix.

---

## 🚀 How to Install and Run (Windows)

### 🧱 Step 1: Get the project

```powershell
git clone https://github.com/hitgalani/EV_Battery_Health_Prediction.git
cd EV_Battery_Health_Prediction
```

### 🐍 Step 2: Make a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 📦 Step 3: Install the libraries

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### ▶️ Step 4: Run the project

Run from the **project root** (the folder that contains `Dataset`, `Python_Code`, and `Reports`), because the script uses project-relative paths.

```powershell
python -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

**Without activating the environment:**

```powershell
.\.venv\Scripts\python.exe -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

### 💡 Tips

- 🔒 If PowerShell blocks activation, use the "without activating" command.
- 📂 If your prompt ends with `\Python_Code>`, run `cd ..` first.
- 🖼️ Charts open one at a time. **Close each chart window to continue.**

---

## ⌨️ Using the Prediction Prompts

When the program asks, type **five numbers**:

| 💬 Prompt | 📝 Meaning | 📏 Unit | 🌟 Example |
|---|---|---|---|
| Vehicle age | How old the vehicle is | months | 48 |
| Total charging cycles | Charge cycles so far | cycles | 400 |
| Average temperature | Typical operating temperature | °C | 25 |
| Fast-charge ratio | Share of fast charging | 0 to 1 | 0.49 |
| Average discharge rate | Typical discharge speed | C-rate | 1.49 |

### 🖨️ Output Format

```text
Battery status: <Healthy or Not Healthy>
Predicted State of Health: <number> %
Probability of Healthy: <number> %
Advice: <example advice text>
Learning project on synthetic data; not for real safety decisions.
```

### 🌟 Example (previously recorded, not a new run)

Inputs **48, 400, 25, 0.49, 1.49** gave: *Healthy*, SoH *96.34 %*, probability *100.00 %*, advice *keep using*.

> 🎓 This only shows the output format. It is **not** real battery advice.

---

## 🤝 Teamwork Guide: Two People, One Project

Two people, one project, zero chaos! 🎯 Here is how **Hit** and **Goti** work together.

### 🎭 Who Does What

| | 👑 Hit Galani (Head) | 🤝 Goti Nisarg (Co-Developer) |
|---|---|---|
| **GitHub** | `hitgalani` | `GotiNIsarg` |
| **Owns** | Repository, `main` branch, releases | Feature and fix branches |
| **Manages** | Git, GitHub, merging, project direction | Code fixes, extra updates |
| **Improves** | Core code and overall plan | Layout, looks, extra guides, instructions |
| **Reviews** | Goti's pull requests | Hit's pull requests (friendly second look) |

### 🌟 Golden Rules

1. 🛑 **Nobody edits `main` directly.** Always use a branch and a Pull Request (PR).
2. 💬 **Talk before big changes.** Tell the other person what you plan to change.
3. 🧪 **Run the project before you push.** If it breaks on your computer, it will break on theirs.
4. 🧹 **Small changes beat giant changes.** One idea per branch.
5. 🙏 **Be kind in reviews.** Say what is good first, then what could be better.
6. 📝 **Update the Changelog** for every merged change.
7. 🔒 **Never push** `.venv/`, passwords, or personal files.

### 📋 Daily Teamwork Routine

```text
☀️ Start of work:
   1. git checkout main
   2. git pull origin main          ← get the latest version
   3. git checkout -b my-new-branch ← start your own branch

🛠️ While working:
   4. Make a small change
   5. Run the project and test it
   6. git add .
   7. git commit -m "✨ Short, clear message"

🌙 End of work:
   8. git push -u origin my-new-branch
   9. Open a Pull Request on GitHub
  10. Ask your teammate to review
```

### 🔀 Who Merges?

- 👑 **Hit** is the final merger into `main`.
- If Hit is busy, Goti may merge **only after Hit says OK** in the PR comments.

### 📣 How We Communicate

| Situation | What to do |
|---|---|
| New idea 💡 | Open a GitHub **Issue** |
| Found a bug 🐛 | Open an Issue with the label `bug` |
| Question ❓ | Comment on the Issue or PR |
| Big change 🏗️ | Chat first, then make a branch |
| Stuck 😵 | Ask! No one is judged here. |

### 🧩 Avoiding Merge Conflicts

Merge conflicts happen when both people edit the **same lines**. To avoid them:

- 🗂️ **Split areas:** Hit works on core code (`Python_Code/`); Goti works on layout, looks, and docs (`README.md`, `Learning_Website/`, `Reports/` styling). If you must touch the same file, tell each other first.
- 🔄 **Pull often** from `main`.
- ✂️ **Keep branches short-lived** (a few days at most).

**If a conflict happens:**

```powershell
git checkout main
git pull origin main
git checkout my-branch
git merge main
# Open the files with conflict markers (<<<<<<<, =======, >>>>>>>)
# Choose the right lines, remove the markers, then:
git add .
git commit -m "🔀 Resolve merge conflict"
```

### ✅ Pull Request Checklist

Copy this into every PR:

```markdown
## What changed?
(Short description)

## Why?
(The reason)

## Checklist
- [ ] I ran `Python_Code/project.py` and it works
- [ ] I did not push `.venv/` or private files
- [ ] I updated the Changelog
- [ ] I updated the README if needed
- [ ] Text is friendly and easy to read
```

---

## 🎨 Layout and Looks Guidelines

*(Goti Nisarg leads this area. Hit approves the final look.)* 🎨

### ✍️ Writing Style
- 😊 Use **short, simple sentences** that a school student can read.
- 🌍 Avoid slang that people in other countries may not understand.
- 🔤 Explain every technical word the **first time** it appears (or add it to the [Glossary](#-mini-glossary)).
- 📏 Use real units (%, °C, months) and always write them.

### 🧱 Structure Style
- 📌 Use headings in order: `#` then `##` then `###`.
- 📊 Use **tables** for lists of facts and **bullets** for steps.
- 🧾 Put code in code blocks with the language name (` ```powershell `, ` ```python `).
- 🖼️ Give every image a short description (alt text) for accessibility.

### 😀 Emoji Style
- ✅ One emoji at the start of a heading or key bullet is nice.
- ❌ Do not put many emojis in a row (🎉🎉🎉🎉🎉).
- 🎯 Keep the same emoji for the same meaning:

| Emoji | Always means |
|---|---|
| ⚠️ | Warning |
| 💡 | Tip |
| ✅ | Done / good |
| ❌ | Not allowed / bad |
| 🧪 | Synthetic data or testing |
| 📌 | Important note |

### 🖼️ Chart Style
- 🏷️ Every chart needs a **title**, **axis labels**, and **units**.
- 🌈 Use colors that are easy to see (also for colour-blind people).
- 💾 Save charts as `.png` in `Reports/` with clear names like `soh_vs_cycles.png`.

### 🌐 Learning Website Style
- 📱 Make it readable on a phone and a laptop.
- 🔡 Use big, clear text and plenty of space.
- 🖼️ Keep images inside `Learning_Website/images/`.

---

## 💻 Code Guidelines

### 🐍 Python Style

| Rule | Example |
|---|---|
| Use clear names | `feature_columns`, not `fc` |
| Use `snake_case` for variables | `new_battery` |
| One idea per code block | Keep sections short |
| Add comments that explain **why**, not just what | `# Scale inputs so cycles do not dominate` |
| Keep paths project-relative | Never hard-code `C:\Users\...` |
| Do not change the CSV file | Work on copies of the data |

### 🔒 Protect the Honesty of the Project
- 🚫 Never print or save made-up scores. Only show numbers the code really computed.
- 🔁 If you change the code or data, **re-run** and update the Results section with real new numbers.
- 🧪 Always keep the "synthetic data, learning only" message in the program output.
- 🛡️ Add input checks (for example, reject negative cycles or a fast-charge ratio above 1) **only when the code really does it**, and document it.

### 📦 Dependencies
- ➕ Added a new library? Update `requirements.txt` with the exact version.
- 📣 Tell your teammate in the PR.

---

## 🌿 Git and GitHub Workflow

### 🌳 Branch Names

| Type | Pattern | Example |
|---|---|---|
| ✨ New feature | `feature/short-name` | `feature/input-validation` |
| 🐛 Bug fix | `fix/short-name` | `fix/chart-labels` |
| 📝 Docs / looks | `docs/short-name` | `docs/readme-layout` |
| 🎨 Style | `style/short-name` | `style/learning-website` |

### 💬 Commit Message Style

| Emoji | Use for | Example |
|---|---|---|
| ✨ | New feature | `✨ Add input range checks` |
| 🐛 | Bug fix | `🐛 Fix wrong axis label` |
| 📝 | Docs | `📝 Update README results` |
| 🎨 | Looks / layout | `🎨 Improve website spacing` |
| ♻️ | Cleanup | `♻️ Simplify EDA section` |
| 🔧 | Config | `🔧 Update requirements.txt` |
| 🎉 | Milestone | `🎉 Release v1.1` |

### 🏷️ Issue Labels (Suggested)

`bug` 🐛 · `idea` 💡 · `docs` 📝 · `looks` 🎨 · `good first task` 🌱 · `help wanted` 🙋 · `question` ❓

### 🙈 What `.gitignore` Should Cover

```text
.venv/
__pycache__/
*.pyc
.vscode/
```

*(Check the real `.gitignore` before changing it.)*

### 🔖 Versions (Simple Plan)

| Version | Meaning |
|---|---|
| `v1.0` | First complete, working learning project |
| `v1.1` | Small fixes and layout improvements |
| `v1.2` | Extra guides and instructions |
| `v2.0` | Big change (for example, a better dataset) |

Hit creates the version tags on GitHub. 🏷️

---

## 🧯 Common Problems and Fixes

| 😵 Problem | 💊 Fix |
|---|---|
| PowerShell blocks `Activate.ps1` | Use `.\.venv\Scripts\python.exe ...` directly |
| `FileNotFoundError` for the CSV | You are in the wrong folder. Run `cd` to the **project root** |
| Prompt shows `\Python_Code>` | Run `cd ..` and try again |
| Charts do not continue | **Close the chart window** |
| `ModuleNotFoundError` | Activate `.venv` and run `python -m pip install -r requirements.txt` |
| Sweetviz warning about `pkg_resources` | Use the run command with the `-W ignore:pkg_resources...` option |
| Different numbers than the README | Data, code, or library versions changed. Re-run and update the Results |
| Git says "your branch is behind" | Run `git pull origin main` |
| Git conflict | See [Avoiding Merge Conflicts](#-avoiding-merge-conflicts) |

---

## 🎉 Fun Corner

### 🧠 Fun Facts
- 🔋 Many EV batteries are designed to last for **many years**, but real life depends on heat, charging habits, and age.
- 🌡️ Hot weather is not a battery's best friend.
- ⚡ Fast charging is handy, but doing it all the time can add wear. That is why we have a `Fast_Charge_Ratio` column!
- 🐘 Our dataset has 9,996 Healthy batteries and only 4 that need replacing. That is like finding **4 red balloons in a room of 10,000 blue ones**. 🎈

### 🏅 Mini Challenges (Try Them!)

| Level | Challenge |
|---|---|
| 🌱 Easy | Run the project and change one input to see how the prediction changes |
| 🌿 Medium | Add a friendly error message when someone types a letter instead of a number |
| 🌳 Hard | Add a new chart and save it in `Reports/` |
| 🚀 Expert | Compare Logistic Regression with another beginner model and explain the difference |

### 🏆 Team Badges (Just for Fun)

- 🥇 **First Merge Champion** (first approved PR)
- 🐛 **Bug Squasher** (fix a bug)
- 🎨 **Layout Wizard** (big looks improvement)
- 📚 **Guide Master** (write a new guide)
- 🧪 **Honest Scientist** (catch a misleading result and explain it)

### 🎮 Try This "What If?" Game
Enter two batteries: one **young with few cycles**, one **old with many cycles**. Which one does the model say is healthier? Is it what you expected? 🤔

---

## 🗺️ Roadmap (Future Ideas)

### 🟢 Next Small Steps
- [ ] ✅ Add friendly input checks (and document them honestly)
- [ ] 📝 Write a short `PROJECT_GUIDE.md` (the old README links to it)
- [ ] 🗂️ Explain what `main.py`, `main2.py`, and `Main/` are, or tidy them up
- [ ] 🎨 Improve the Learning Website look
- [ ] 📊 Add clearer chart titles and labels

### 🟡 Medium Steps
- [ ] 🧪 Find a **licensed, teacher-approved** dataset with enough examples of each status
- [ ] ⚖️ Handle class imbalance in a careful, explained way
- [ ] 📈 Add more evaluation views (for example, a precision-recall idea explained simply)

### 🔴 Big Dreams
- [ ] 🔮 Explore **remaining useful life** prediction if suitable target data exists
- [ ] 🌐 Build a simple web app **only after** input validation and model validation
- [ ] 📡 Consider real-time monitoring **only with validated sensors and safety review**

---

## 📝 Changelog and Update Log

> 📌 Add a new line here for **every merged change**. Newest on top.

| 📅 Date | 🔖 Version | 👤 By | 📝 What changed |
|---|---|---|---|
| YYYY-MM-DD | v1.x | Hit / Goti | *Describe the change here* |
| YYYY-MM-DD | v1.0 | Hit Galani | 🎉 First working version: data cleaning, EDA, 3 models, evaluation, user prediction |

### 🧾 Entry Template

```markdown
| 2026-MM-DD | v1.1 | Goti Nisarg | 🎨 Improved README layout and added teamwork guide |
```

### 🎁 Small Surprise Updates ("Fun Updates")
Small, happy improvements are welcome! Examples: a nicer chart color, a friendlier message, a new emoji set, or a fun fact in the output. Log them here too. 😄

---

## 📚 Learning Log

Both teammates write what they learned. This keeps the project a **learning** project! 🌱

| 📅 Date | 👤 Who | 💡 What I learned |
|---|---|---|
| YYYY-MM-DD | Hit | *Example: Accuracy can be misleading with unbalanced data* |
| YYYY-MM-DD | Goti | *Example: How to open a Pull Request and resolve a conflict* |

---

## 📖 Mini Glossary

| Word | Easy meaning |
|---|---|
| **EV** | Electric vehicle 🚗⚡ |
| **SoH** | State of Health. How healthy a battery is compared with new |
| **Synthetic data** | Data made by a computer, not measured from real things |
| **Feature** | An input the model uses (like charging cycles) |
| **Target** | The answer we want the model to predict |
| **Regression** | Predicting a **number** |
| **Classification** | Predicting a **category** |
| **Train set** | Data the model learns from 📖 |
| **Test set** | Unseen data used to check the model 📝 |
| **Cross-validation** | Testing several times on different splits to see if scores are stable |
| **Overfitting** | Learning the training data too well and failing on new data |
| **Target leakage** | Giving the model a clue that reveals the answer |
| **EDA** | Exploratory Data Analysis. Looking at data before modeling |
| **Outlier** | A value far from the rest |
| **IQR** | Q3 − Q1, used to spot outliers |
| **Scaling** | Putting numbers on a similar size range |
| **Class imbalance** | One category is far more common than the other |
| **Git** | A tool that tracks changes in files |
| **GitHub** | A website that stores Git projects online |
| **Branch** | A separate copy of work where you can make changes safely |
| **Pull Request (PR)** | A request to add your branch's changes to `main` |
| **Merge conflict** | Two people changed the same lines |

---

## ❓ FAQ

**❓ Can I use this to check my real car battery?**
🚫 No. It uses synthetic data and is for learning only.

**❓ Why is accuracy so high but the model still not great?**
⚖️ Because almost every row is Healthy. See [Class Imbalance Warning](#️-class-imbalance-warning).

**❓ Can a beginner join this project?**
🌱 Yes! Start with the "Easy" mini challenge and open an Issue to say hello.

**❓ Why does the project use only five inputs?**
🎯 To keep it simple and avoid target leakage. More inputs may be tested later.

**❓ Who do I ask for help?**
💬 Open an Issue on GitHub. Hit (`hitgalani`) manages the project, and Goti (`GotiNIsarg`) helps with fixes, layout, and guides.

---

## 🙏 Credits and Thanks

- 👑 **Hit Galani** ([@hitgalani](https://github.com/hitgalani)): project head, Git and GitHub manager
- 🤝 **Goti Nisarg** ([@GotiNIsarg](https://github.com/GotiNIsarg)): co-developer, code fixes, layout, looks, and guides
- 📦 Dataset: *Electric Vehicle (EV) Battery Degradation and Charge* on [Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge)
- 🛠️ Open-source tools: Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Sweetviz
- 🎓 Our teachers and the open learning community 🌍

> 📜 **License:** *Not yet chosen.* Hit should decide and add a `LICENSE` file before sharing widely. Also check the Kaggle dataset's own license.

---

## ✅ Things to Double-Check Before Presenting

This README was built from the recorded project notes. The Python files (`project.py`, `main.py`, `main2.py`) and the CSV were **not re-inspected**, and the code was **not re-run**. Please verify these items against your real files:

- [ ] 🔢 Dataset size (10,000 × 13) and class counts (9,996 Healthy / 4 Replace Required)
- [ ] 🧹 Exact cleaning behavior (duplicates, missing values)
- [ ] 🎯 Whether the classification split is **stratified**
- [ ] 🔒 Whether input validation exists (and the allowed ranges)
- [ ] 📏 Real min and max of each input column in the CSV
- [ ] 📄 What `main.py`, `main2.py`, and `Main/` are for
- [ ] 📖 Whether `PROJECT_GUIDE.md` exists, or remove the link from the old README
- [ ] 💻 The run commands match your real setup
- [ ] 📊 Results still match after a fresh run (update the tables if not)
- [ ] 🧪 Teacher approval for using a synthetic dataset

---

<div align="center">

### 💚 Thank you for visiting! Keep learning, keep building, keep smiling! 😊🔋

**Made with ❤️ by Hit Galani 👑 and Goti Nisarg 🤝**

*Learning project only. Not for real battery safety decisions.* 🧪

</div>
