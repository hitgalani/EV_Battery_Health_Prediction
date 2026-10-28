# EV Battery Health Prediction — Beginner Project Guide

This guide explains the project from the Python file `Python_Code/project.py` and the CSV `Dataset/EV_Battery_Dataset.csv`. It is written for a student who is learning Python and machine learning for the first time.

> **Use the code as the source of truth.** The project has more than one Python file, and they may not implement exactly the same workflow. This guide describes `Python_Code/project.py`, which is the main project script discussed here. Run it from the project root using the command in [How to run the project](#20-deployment-how-to-run-the-project).
>
> **Important difference from the requested outline:** `project.py` trains Simple Linear Regression, Multiple Linear Regression, and Logistic Regression. It does **not** train Ridge Regression. Do not describe Ridge as a model used by this script.

## 1. Project introduction

This project uses measurements about an electric-vehicle battery to make two educational predictions:

1. Estimate the battery's **State of Health (SoH)** as a percentage.
2. Predict the battery-status category recorded in the dataset.

The program reads existing example records, explores and prepares them, trains machine-learning models, checks model performance, and asks the user to enter measurements for a new prediction.

The Python script calls the data synthetic and warns that it is **not a battery safety test**. Its predictions must not be used to decide whether a real battery is safe or needs replacement.

## 2. Project objectives

- Read and inspect the supplied EV battery CSV file.
- Check for missing values and duplicate rows.
- Explore the data with summary values and charts.
- Predict `SoH_Percent`, a numeric target, using regression.
- Predict the recorded battery status using classification.
- Evaluate the models with test data and cross-validation.
- Accept five battery measurements and display an educational prediction.

## 3. Python libraries

These are the six external libraries imported by `project.py`.

| Library | Simple explanation | How this project uses it |
|---|---|---|
| `pandas` | Works with table-shaped data, like a spreadsheet. | Reads the CSV, selects columns, checks missing or duplicate values, and creates tables for charts and user input. |
| `numpy` | Provides numeric calculations and arrays. | Calculates the square root for RMSE and clips the user-input SoH prediction to 0–100. |
| `matplotlib` | Makes charts and saves them as image files. | Creates figures, tables, bar charts, pie charts, and the sample-prediction chart. |
| `seaborn` | Provides convenient statistical charts based on Matplotlib. | Draws the histograms and correlation and confusion-matrix heatmaps. |
| `scikit-learn` | Provides machine-learning models, data splitting, and evaluation tools. | Trains regression and classification models, splits data, runs cross-validation, and calculates metrics. |
| `sweetviz` | Creates an automated exploratory-data-analysis report. | Saves `Reports/Sweetviz_EDA_Report.html`. |

`input()`, `print()`, and file-path handling are built into Python; they are not extra external libraries.

## 4. Dataset explanation

The program reads `Dataset/EV_Battery_Dataset.csv`. The supplied file has **10,000 rows and 13 columns**. Each row represents one battery/vehicle record. A column contains one type of information about those records.

| CSV column | Meaning | Type or unit |
|---|---|---|
| `Vehicle_ID` | Record identifier | Text ID; not used as a model input |
| `Car_Model` | Vehicle model | Category; not used as a model input |
| `Battery_Type` | Battery chemistry/type | Category; not used as a model input |
| `Battery_Capacity_kWh` | Battery capacity | kWh; not used as a model input |
| `Vehicle_Age_Months` | Vehicle age | Months; model input |
| `Total_Charging_Cycles` | Total recorded charging cycles | Cycles; model input |
| `Avg_Temperature_C` | Average temperature | Degrees Celsius; model input |
| `Fast_Charge_Ratio` | Share of charging that was fast charging | Ratio from 0 to 1 in the supplied data; model input |
| `Avg_Discharge_Rate_C` | Average battery discharge rate | C-rate; model input |
| `Driving_Style` | Driving-style category | Category; not used as a model input |
| `Internal_Resistance_Ohm` | Battery internal resistance | Ohms; not used as a model input |
| `SoH_Percent` | Battery State of Health | Percent; regression target |
| `Battery_Status` | Status label supplied in the dataset | Category; classification target |

The script maps the recorded status to a number for classification: `Healthy` becomes `1`; every other status becomes `0`. The dataset labels include `Healthy` and `Replace Required`. The program later prints the readable prediction as `Healthy` or `Not Healthy`, so its displayed text is not an exact copy of the dataset's second label.

The current CSV was checked while preparing this guide: it has **9,996 `Healthy` records and 4 `Replace Required` records**. This imbalance is important: there are very few examples of the second status, so classification performance for that group is uncertain.

## 5. Complete project workflow

The numbered stages below follow the order and stage labels in `project.py`.

1. **Problem definition:** print the prediction goals and safety limitation.
2. **Data collection:** read the CSV, display basic information, and save a summary chart.
3. **Data preparation:** count potential feature outliers, remove duplicate rows, and remove rows missing required fields.
4. **Exploratory data analysis (EDA):** print SoH statistics, save Sweetviz's HTML report, and create charts.
5. **Feature selection:** select five inputs and define SoH and battery-status targets.
6. **Train/test split:** reserve data for testing and keep a separate stratified split for status classification.
7. **Model training:** train Simple Linear Regression, Multiple Linear Regression, and Logistic Regression.
8. **Model evaluation:** calculate regression and classification measures, save results, and run five-fold cross-validation.
9. **Prediction:** compare a Multiple Linear Regression prediction with an actual SoH from one test row.
10. **Deployment with user input:** ask for five values and predict SoH, status, probability, and advice.
11. **Monitoring:** print reminders about checking future results and getting teacher approval.

## 6. Data collection

```python
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
```

- `pd.read_csv(...)` opens the CSV at the given project-relative path.
- The result is a Pandas DataFrame (a table) named `df`.
- The path works when the program is launched from the project root, which contains the `Dataset` folder.
- The code then prints the shape, column names, first five rows, data types, numeric statistics, and battery-status counts.

If the CSV is not at that exact path relative to the current working directory, Python reports a file-not-found error.

## 7. Data inspection

The program uses these checks and summaries:

```python
print("Dataset shape (rows, columns):", df.shape)
print("Column names:", list(df.columns))
print(df.head())
df.info()
print(df.describe())
print(df["Battery_Status"].value_counts())
```

- `df.shape` returns the number of rows and columns. For the supplied CSV it is `(10000, 13)`.
- `list(df.columns)` displays the names used to select the model fields.
- `df.head()` displays the first five records.
- `df.info()` prints each column's data type and non-missing count.
- `df.describe()` summarizes numeric columns, including count, average, spread, minimum, and maximum.
- `value_counts()` counts each battery-status label.

The data-preparation section also measures missing values in the required model columns and duplicate rows. The supplied CSV was checked for this guide and has zero missing cells and zero duplicate rows.

## 8. Data cleaning

```python
df = df.drop_duplicates()
df = df.dropna(subset=required_columns)
```

- `drop_duplicates()` removes rows that are exact duplicates of other rows.
- `dropna(subset=required_columns)` removes a row if one or more specified input or target fields are missing.
- The script does **not** fill missing values with a mean, median, or mode. It removes incomplete rows in required fields.
- The code counts values outside IQR limits for each of the five input features. IQR means the middle half of the values; the limits are `Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR`.
- The script reports those potential outliers but explicitly keeps them. They are counted, not removed.

The code records before-and-after checks in a table and saves it to `Reports/data_cleaning.png`.

## 9. Exploratory Data Analysis (EDA)

EDA means looking at the data using summaries and charts before relying on model results. The script creates the following charts and files.

| Chart/report | Saved file | What it shows and why it helps |
|---|---|---|
| Dataset summary table | `Reports/dataset_summary.png` | Basic row, column, missing, duplicate, and label counts. |
| Data preparation table | `Reports/data_cleaning.png` | Before-and-after cleaning counts and counted input outliers. |
| Sweetviz report | `Reports/Sweetviz_EDA_Report.html` | Automated summaries and relationships for the data. A temporary Boolean `Healthy` column is used as the report target; the original dataframe is not given that extra column. |
| SoH histogram | `Reports/soh_histogram.png` | Shows how often SoH values fall into different ranges. |
| SoH box plot | `Reports/soh_boxplot.png` | Shows the middle spread and points far from the typical range. |
| Charging-cycles scatter plot | `Reports/cycles_vs_soh.png` | Shows how SoH varies with charging cycles, one record per point. |
| Correlation heatmap | `Reports/correlation_heatmap.png` | Shows numeric correlation between inputs and SoH. Correlation is a relationship, not proof that one value causes another. |
| Battery-status bar chart | `Reports/battery_status_counts.png` | Compares the number of records in each status category. |
| Battery-status pie chart | `Reports/battery_status_pie.png` | Shows each status category as a share of the dataset. |
| Battery-type bar chart | `Reports/battery_type_bar.png` | Counts records by battery type. |
| Feature/target table | `Reports/feature_split.png` | Summarizes the model inputs and targets. |
| Simple regression line | `Reports/simple_regression_line.png` | Shows the fitted charging-cycle-only line against the dataset points. |
| Regression results table | `Reports/model_results.png` | Displays test-set regression measures. |
| Train/test score table | `Reports/train_test_scores.png` | Compares model scores on training and test rows. |
| Classification results table | `Reports/classification_metrics.png` | Displays classification measures. |
| Confusion matrix heatmap | `Reports/confusion_matrix.png` | Compares actual and predicted status labels. |
| Cross-validation table | `Reports/cross_validation.png` | Shows scores for each of five data folds. |
| Test-row prediction chart | `Reports/sample_prediction.png` | Compares actual and predicted SoH for one held-out row. |
| User-input prediction chart | `Reports/deployment_prediction.png` | Shows the SoH predicted for the user's entered values. |

Charts are saved and then displayed. Close each chart window to let the script continue.

## 10. Feature selection

The script uses these five features (input columns):

| Feature | Meaning | Unit |
|---|---|---|
| `Vehicle_Age_Months` | Age of the vehicle | Months |
| `Total_Charging_Cycles` | Total number of charging cycles | Cycles |
| `Avg_Temperature_C` | Average recorded temperature | °C |
| `Fast_Charge_Ratio` | Fraction of charging that is fast charging | Ratio from 0 to 1 in the supplied data |
| `Avg_Discharge_Rate_C` | Average discharge rate | C-rate |

The targets are:

- **Regression target:** `SoH_Percent`, a numeric percentage.
- **Classification target:** `Battery_Status`, converted to `1` for `Healthy` and `0` for anything else.

The code uses the same five inputs for both tasks. It does not include `SoH_Percent` as a classification input, which avoids giving the classifier the answer it is meant to predict (target leakage).

## 11. Train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

- **Training data** is the part the model learns from.
- **Test data** is held back to check predictions on records not used for fitting.
- `test_size=0.20` requests 20% for testing and leaves 80% for training.
- `random_state=42` makes the random split repeatable.
- For classification, the script makes another split with `stratify=y_class`. This tries to keep the very uneven class proportions represented in both parts.

With 10,000 clean rows, the split uses approximately 8,000 training rows and 2,000 test rows for each task. The actual row allocation of the rare status should be read from the program's printed counts, not guessed.

## 12. Machine-learning models

### Simple Linear Regression

`model1` uses only `Total_Charging_Cycles` to estimate `SoH_Percent`. “Simple” means one input feature. The plotted line helps show the fitted relationship between cycles and SoH.

### Multiple Linear Regression

`model2` uses all five input features to estimate `SoH_Percent`. “Multiple” means several input features are used together. The script also uses this model for the held-out sample comparison and user-input SoH prediction.

### Ridge Regression

**Ridge Regression is not present in `Python_Code/project.py`.** The script imports `LinearRegression` and `LogisticRegression`; it does not import or fit `Ridge`. It would be inaccurate to claim this code trains or evaluates a Ridge model.

### Logistic Regression

`model3` uses the five input features to predict the binary status value (`1` or `0`). It is a classification model even though “regression” appears in its name.

The model is put in a `Pipeline` with `StandardScaler`. Scaling puts feature measurements on comparable numeric scales for learning. `class_weight="balanced"` asks the classifier to pay more attention to the rare class during training; it cannot replace having enough rare-class examples.

## 13. Model training

```python
model1 = LinearRegression()
model1.fit(X_train[["Total_Charging_Cycles"]], y_train)

model2 = LinearRegression()
model2.fit(X_train, y_train)

model3 = Pipeline([
    ("scale", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=5000, class_weight="balanced")),
])
model3.fit(X_train_class, y_train_class)
```

- Creating a model selects the learning method and its options.
- `.fit(inputs, answers)` trains the model. It looks at the training inputs and their known targets and learns a pattern.
- `model1` receives one column; `model2` receives five columns; `model3` receives the five classification inputs and numeric status targets.
- `max_iter=5000` sets the maximum number of learning iterations for Logistic Regression.
- The scaling step is fitted as part of the pipeline using training data and then applied consistently when the pipeline predicts.

## 14. Model evaluation

The script calculates regression measures for `model1` and `model2`, and classification measures for `model3`.

### Regression measures

| Measure | Beginner explanation |
|---|---|
| MAE | Mean Absolute Error: average size of the prediction error, ignoring whether it is above or below the real value. For SoH, it is in percentage points. Lower is generally better. |
| MSE | Mean Squared Error: average squared error. Large mistakes count especially strongly. Lower is generally better. |
| RMSE | Root Mean Squared Error: square root of MSE, so the scale returns to SoH percentage points. Lower is generally better. The script calculates it with `np.sqrt(mse)`. |
| R² | A score comparing the model with a simple baseline. A value closer to 1 generally means a better fit on the evaluated data; it is not a percent-accuracy score. |
| RSS | Residual Sum of Squares: sum of squared prediction errors. Lower means a smaller total squared error on these test records. |

### Classification measures

| Measure | Beginner explanation |
|---|---|
| Accuracy | Share of all test rows whose status was predicted correctly. With a very rare status, high accuracy can hide mistakes on that rare status. |
| Precision (Not Healthy) | Of rows predicted as `Not Healthy` (class 0), how many were actually class 0. |
| Recall (Not Healthy) | Of actual class-0 rows, how many the model found. |
| F1 (Not Healthy) | A combined measure of precision and recall for class 0. |
| Confusion matrix | A table of actual versus predicted classes. This script uses row order `[0, 1]` and column order `[0, 1]`: Not Healthy, then Healthy. |

The supplied dataset has only four `Replace Required` rows. The code itself warns that classification scores are uncertain because of this imbalance. Do not treat accuracy alone as proof that the status model is reliable.

## 15. Cross-validation

`cross_val_score()` divides data into five folds. It trains and evaluates a model repeatedly, each time holding a different fold out for evaluation. This gives more than one evaluation split and shows how scores vary.

The script reports:

- Five-fold R² scores for Simple Linear Regression.
- Five-fold R² scores for Multiple Linear Regression.
- Five-fold accuracy scores for Logistic Regression.

The code uses shuffled `KFold` splits for these calculations. It does not use stratified folds for the rare classification label, so the four rare examples may be unevenly distributed across folds. Interpret classification cross-validation accuracy carefully.

## 16. Sample prediction

```python
sample = X_test.iloc[[0]]
sample_actual = y_test.iloc[0]
sample_prediction = model2.predict(sample)[0]
```

- `X_test.iloc[[0]]` selects the first test row as a one-row table. The double brackets keep it in the two-dimensional table shape expected by the model.
- `y_test.iloc[0]` gets that row's known SoH.
- `model2.predict(sample)` estimates SoH from its five input features.
- The script prints the actual and predicted SoH and saves a bar chart to `Reports/sample_prediction.png`.

This is a held-out example from the supplied data, not a prediction for the user's new battery.

## 17. User input section — all five measurements

The script requests these fields in this order:

| Prompt | Column | Meaning and unit | Example values |
|---|---|---|---|
| `Vehicle age (months)` | `Vehicle_Age_Months` | Vehicle age in months. Type a number, such as `12`, `48`, or `84`. The supplied data ranges from 1 to 96 months. |
| `Total charging cycles (cycles)` | `Total_Charging_Cycles` | Number of charging cycles. Type a number, such as `100`, `400`, or `900`. The supplied data ranges from 4 to 1,436 cycles. |
| `Average temperature (°C)` | `Avg_Temperature_C` | Average recorded temperature in degrees Celsius. Type a number, such as `10.5`, `25`, or `40`. The supplied data ranges from -12.9°C to 54.3°C. |
| `Fast-charge ratio (0 to 1)` | `Fast_Charge_Ratio` | Fraction of charging that was fast charging. Type a decimal from 0 to 1, such as `0.25`, `0.50`, or `0.80`. For example, `0.25` represents a ratio of 25%. The supplied data ranges from 0 to 1. |
| `Average discharge rate (C-rate)` | `Avg_Discharge_Rate_C` | Average battery discharge rate in C-rate. Type a number, such as `0.8`, `1.5`, or `2.2`. The supplied data ranges from 0.5 to 2.5 C-rate. |

These example values are examples for entering numeric inputs, not recommended or guaranteed safe operating values.

**Supported ranges need careful wording:** the ranges above are the minimum and maximum values observed in the supplied CSV, not validation rules in the Python code. The program does not check whether a user's values fall in these ranges. It only converts the typed text to a number and sends it to the models. The prompt says the fast-charge ratio is 0–1, but there is no code enforcing that range.

Every feature is passed to the trained model in the same column order as `feature_columns`. The model uses all five values together; the script does not show a separate numerical contribution for each one.

## 18. Prediction calculation and final user-input code

The core code at the end of `project.py` is:

```python
age_input = float(input("Vehicle age (months): "))
cycles_input = float(input("Total charging cycles (cycles): "))
temperature_input = float(input("Average temperature (°C): "))
charge_input = float(input("Fast-charge ratio (0 to 1): "))
discharge_input = float(input("Average discharge rate (C-rate): "))

new_battery = pd.DataFrame(
    [[age_input, cycles_input, temperature_input, charge_input, discharge_input]],
    columns=feature_columns,
)

status_prediction = int(model3.predict(new_battery)[0])
healthy_probability = model3.predict_proba(new_battery)[0][1] * 100
soh_prediction = float(np.clip(model2.predict(new_battery)[0], 0, 100))
```

In order:

1. Each `input("...")` displays a prompt and waits for the user to type text and press Enter.
2. `float(...)` converts that text to a decimal number. Typing `twenty` instead of a number causes a `ValueError`; the program does not catch that error.
3. `pd.DataFrame(...)` arranges the five values into a one-row table. `columns=feature_columns` labels each value with the exact feature name expected by the models.
4. `model3.predict(new_battery)` predicts a class. `[0]` selects the first result because the table contains one row. `int(...)` converts that class to an integer.
5. `model3.predict_proba(new_battery)[0][1]` selects the estimated probability for class `1`, which the code defines as Healthy. Multiplying by 100 displays it as a percentage.
6. `model2.predict(new_battery)[0]` estimates SoH. `np.clip(value, 0, 100)` limits the displayed estimate to the 0–100 interval. This clipping is only applied to the user's final SoH prediction; it is not applied to the model's evaluation metrics or held-out sample prediction.

The following readable status and advice logic comes immediately after:

```python
if status_prediction == 1:
    battery_status = "Healthy"
else:
    battery_status = "Not Healthy"

if battery_status == "Not Healthy":
    advice = "consider replacement"
elif soh_prediction < 80:
    advice = "check soon"
else:
    advice = "keep using"
```

- `if` checks the predicted class. `1` is displayed as `Healthy`; `0` is displayed as `Not Healthy`.
- The advice is selected using the predicted status first. If status is `Not Healthy`, the advice is `consider replacement`. Otherwise, a SoH below 80 gives `check soon`; all other cases give `keep using`.
- The 80% value is an advice threshold in this code, not a rule used to create the training status label. The model predicts status separately, so status and SoH-based advice can disagree.
- The script prints battery status, SoH, Healthy probability, advice, and a safety disclaimer.
- It then saves and displays `Reports/deployment_prediction.png`, a bar chart of the predicted SoH.

### Hypothetical terminal interaction

The following illustrates the input order and output format only. The values in angle brackets are placeholders; they are **not actual model results**. The charts and evaluation tables appear earlier while the program runs.

```text
Enter five battery measurements from the dataset's named columns.
Vehicle age (months): 48
Total charging cycles (cycles): 400
Average temperature (°C): 25
Fast-charge ratio (0 to 1): 0.50
Average discharge rate (C-rate): 1.5
Battery status: <predicted by the classifier>
Predicted State of Health: <predicted percentage> %
Probability of Healthy: <estimated percentage> %
Advice: <selected by the program>
Learning project on synthetic data; not for real safety decisions.

Stage 11: Monitoring
Compare future predictions with measured outcomes and check errors.
Ask the teacher before using a non-synthetic dataset.
```

## 19. Output explanation

- **Battery status:** classifier output translated to `Healthy` or `Not Healthy`.
- **Predicted State of Health:** regression estimate clipped to 0–100 and printed with two decimal places.
- **Probability of Healthy:** Logistic Regression's estimated probability for class 1, shown as a percentage. It is an estimate from this model and dataset, not a guarantee.
- **Advice:** simple text selected by the `if` conditions described above.

The code prints `Not Healthy`, while the CSV's second category is `Replace Required`. This is a label wording difference in the program, not a new dataset category.

## 20. Deployment: how to run the project

Run the program from the **project root** (the folder containing `Dataset`, `Python_Code`, and `Reports`) because the script uses project-relative paths.

### Windows PowerShell

Create and activate a virtual environment if the project does not already have one:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project's listed dependencies:

```powershell
python -m pip install -r requirements.txt
```

Then run the script from the project root:

```powershell
python -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

If the virtual environment already exists and dependencies are installed:

```powershell
.\.venv\Scripts\python.exe -W error -W "ignore:pkg_resources is deprecated as an API:UserWarning" Python_Code\project.py
```

The dependencies in `requirements.txt` include the six libraries listed earlier and a setuptools compatibility dependency used by Sweetviz. Open the charts as they appear and close each chart window to continue. At the end, enter five numeric values when prompted.

The `Reports` folder receives PNG charts and `Sweetviz_EDA_Report.html`. The script saves these outputs; it does not automatically publish a website or deploy an online service.

## 21. More practical examples

The following are **hypothetical inputs only**. They are not actual predictions. They are within the observed ranges of the supplied dataset.

| Test idea | Vehicle age | Charging cycles | Average temperature | Fast-charge ratio | Discharge rate |
|---|---:|---:|---:|---:|---:|
| Example A | 12 | 100 | 10 | 0.25 | 0.8 |
| Example B | 48 | 400 | 25 | 0.50 | 1.5 |
| Example C | 84 | 900 | 40 | 0.80 | 2.2 |

To try one, run the program and type the five values in the order shown by its prompts. Record the output produced by your own run. Predictions are not provided here because they depend on executing the model and must not be invented.

## 22. Common errors and solutions

| Error or issue | Why it happens | What to check |
|---|---|---|
| `FileNotFoundError` for the CSV | The current working directory is not the project root, or the file/folder name is different. | Run from the folder containing `Dataset`, and confirm the exact name `Dataset/EV_Battery_Dataset.csv`. |
| Error saving a chart or report | The script writes files into `Reports` but does not create that folder itself. | Run from the supplied project, where `Reports` exists, or create a `Reports` folder beside `Dataset` before running. |
| `ModuleNotFoundError` | A required package is not installed in the Python environment running the script. | Activate the project's environment or install the dependencies with `python -m pip install -r requirements.txt`. |
| `ValueError` during input | The response could not be converted by `float()`, for example typing `abc`, leaving the input empty, or using a comma decimal. | Enter one numeric value per prompt, using a period for decimals, such as `0.25`. |
| Unexpected prediction for an unusual input | The code accepts numbers outside the dataset's observed ranges and does not validate them. | Use values in the documented observed ranges for a learning example; remember these are not certified safety limits. |
| Confusing status text | The script displays `Not Healthy`, but the dataset label is `Replace Required`. | Explain that class 0 represents every label other than `Healthy` in the Boolean conversion. |
| High accuracy but missed replacement cases | There are only four `Replace Required` rows, while most rows are Healthy. | Review the confusion matrix and class-specific metrics; do not rely on accuracy alone. |
| Chart window appears to stop the script | The script calls `plt.show()` before proceeding. | Close the current chart window to continue. |

The code does not perform manual text-to-label entry: the user enters five numbers, and the classifier produces the status.

## 23. Project limitations

- The script itself describes the dataset as synthetic; it is for learning, not certified diagnostics.
- Only four supplied rows have the `Replace Required` label. The classifier has very few rare examples to learn from and evaluate.
- The script does not validate user inputs against dataset ranges or physically meaningful constraints.
- The predicted status is a category learned from the supplied labels; it is not a battery inspection.
- The simple advice is hard-coded. In particular, an 80% SoH threshold does not determine the classifier's status.
- The model scores describe this particular dataset and split; they do not prove performance on real vehicles or other datasets.
- Outliers in the five input features are counted but retained.
- Cross-validation uses ordinary shuffled K-fold splits; rare-class examples may not be evenly represented in every fold.
- The user-input SoH is clipped to 0–100, but the model's other reported predictions and test metrics are not clipped.

## 24. Viva preparation: likely questions

1. **What is the aim of your project?**  
   To estimate EV battery State of Health and predict the dataset's battery-status category from five battery measurements.

2. **What does SoH mean?**  
   State of Health is a percentage used here as the numeric battery-health target.

3. **What is the dataset file?**  
   `Dataset/EV_Battery_Dataset.csv`.

4. **What are the five model inputs?**  
   Vehicle age, charging cycles, average temperature, fast-charge ratio, and average discharge rate.

5. **What is the regression target?**  
   `SoH_Percent`.

6. **What is the classification target?**  
   `Battery_Status`, converted to 1 for Healthy and 0 for not Healthy.

7. **Why are there two kinds of models?**  
   Regression predicts a number (SoH); classification predicts a category (battery status).

8. **What is Simple Linear Regression in this project?**  
   It predicts SoH using only charging cycles.

9. **What is Multiple Linear Regression in this project?**  
   It predicts SoH using all five inputs.

10. **Does this project use Ridge Regression?**  
    No. `Python_Code/project.py` uses Simple Linear Regression, Multiple Linear Regression, and Logistic Regression.

11. **What does `fit()` do?**  
    It trains a model using example inputs and their known answers.

12. **Why split training and test data?**  
    To train on one portion and check the model on separate records.

13. **What is cross-validation?**  
    It repeats training and evaluation across several folds to check how scores vary.

14. **What is a confusion matrix?**  
    A table comparing actual classes with predicted classes.

15. **Why is the status prediction uncertain?**  
    The dataset has 9,996 Healthy rows and only 4 Replace Required rows.

16. **What does `Fast_Charge_Ratio = 0.25` mean?**  
    It represents a ratio of 25% in the code's stated 0-to-1 scale.

17. **Does the code check that inputs are in range?**  
    No. It converts the values to numbers but does not check the ranges.

18. **What does `np.clip()` do here?**  
    It limits the final user-input SoH prediction to the range 0–100.

19. **Can this program diagnose a real battery?**  
    No. It is a learning project and not a certified battery safety test.

20. **Where are the project charts saved?**  
    In the `Reports` folder.

## 25. Final project summary

This project loads an EV battery CSV, inspects and cleans required data, explores it with charts, and uses five battery measurements to train models. Simple and Multiple Linear Regression estimate State of Health, while Logistic Regression predicts the recorded battery-status class. The script evaluates its models, performs cross-validation, compares one held-out prediction, and accepts five typed values for a final educational prediction. Its dataset is highly imbalanced and described as synthetic, so results are for learning only and must not be used for real battery safety decisions.
