# EV Battery Health Prediction Using Python and Machine Learning

Complete beginner's guide · College project explanation · Code walkthrough · Practical examples · Viva preparation

This explanation follows the simplified EV battery Python project we prepared earlier, using your teacher's Food Delivery Time Prediction project as a guide to the teaching style.

I'll distinguish between what your code actually does, what the machine learning terms mean, and what still needs to be checked using your real dataset. No actual accuracy scores or prediction results will be invented.

# 1. Project introduction

## What is this project?

Your project is called EV Battery Health Prediction Using Python and Machine Learning.

EV stands for Electric Vehicle. An electric vehicle uses a battery to store the energy needed to run its motor.

Over time, battery health can change because of factors such as:

- The age of the vehicle.
- The number of charging cycles.
- Battery temperature.
- Fast-charging habits.
- The rate at which the battery releases energy.

Your project uses information about these factors to train machine learning models.

The models learn patterns from a dataset containing battery measurements and known battery-health information. After training, the program can estimate the State of Health (SoH) and predict a battery status for a new set of measurements.

## What is machine learning?

Machine learning is a method in which a computer learns patterns from examples instead of following only manually written rules.

For example, imagine you have a dataset with many battery records. Each record contains charging cycles, temperature, and measured State of Health.

A machine learning model examines the relationships between the inputs and the known answers. It then uses those learned relationships to estimate an answer for a new record.

The answer is a prediction, not a guarantee.

## Why is this project useful?

The project demonstrates how machine learning can be used to:

- Study battery measurements.
- Understand patterns in battery health.
- Estimate battery State of Health.
- Classify batteries into health-status categories.
- Compare different prediction models.
- Generate charts and evaluation reports.

This is an educational project. It is not a certified battery diagnostic system.

# 2. Project objectives

Your program has two main prediction tasks.

Objective 1: Predict State of Health (SoH)

## Example: 86%

This is an illustrative value, not an actual prediction from your dataset.

The regression models estimate SoH as a numerical percentage between 0 and 100 in the displayed prediction.

Objective 2: Predict battery status

## Healthy / Replace Required

The classification model predicts one of the two status categories expected by the code.

These are separate tasks:

- Regression predicts a number.
- Classification predicts a category.

The regression model does not directly choose the battery status. The classification model does that separately.

# 3. Python libraries

Your project uses these six external libraries.

| Library        | Simple explanation                           | How your project uses it                                                          |
| -------------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| `pandas`       | Works with data tables.                      | Loads the CSV, cleans rows, selects columns, and creates DataFrames.              |
| `numpy`        | Performs numerical calculations.             | Calculates square roots, handles numerical arrays, and limits displayed SoH.      |
| `matplotlib`   | Draws charts.                                | Creates histograms, scatter plots, box plots, and prediction charts.              |
| `seaborn`      | Helps create statistical charts.             | Creates correlation heatmaps and confusion-matrix visualizations.                 |
| `scikit-learn` | Provides machine learning tools.             | Splits data, trains models, predicts outcomes, and calculates evaluation metrics. |
| `sweetviz`     | Generates an automated data-analysis report. | Creates an HTML report describing the dataset.                                    |

## Why are these six libraries needed?

### `pandas`

Your dataset is a CSV file. Pandas loads it into a table called a DataFrame.

```
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
```

Here, `df` is the variable that stores the dataset in table form.

### `numpy`

```
np.sqrt(mse)
```

This calculates the square root of the MSE value to obtain RMSE.

Your program also uses:

```
np.clip(value, 0, 100)
```

This limits the displayed SoH value to the range 0–100.

### `matplotlib`

```
plt.figure(figsize=(8, 5))plt.hist(df["SoH_Percent"], bins=20)
```

This creates a histogram showing how State of Health values are distributed.

### `seaborn`

```
sns.heatmap(df[features + ["SoH_Percent"]].corr(),            annot=True, cmap="coolwarm")
```

This draws a correlation heatmap showing the relationships between the numerical columns.

### `scikit-learn`

Your project imports several tools from scikit-learn, including:

```
from sklearn.model_selection import train_test_splitfrom sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
```

These tools split the dataset and build prediction models.

`Pipeline`, `SimpleImputer`, and `StandardScaler` are also components of scikit-learn. They are not additional external libraries.

### `sweetviz`

```
report = sv.analyze(df)report.show_html("Reports/Sweetviz_EDA_Report.html",                 open_browser=False)
```

This creates an HTML report for inspecting the dataset.

Installation command:

```
pip install pandas numpy matplotlib seaborn scikit-learn sweetviz
```

You do not need to install each scikit-learn component separately.

# 4. Dataset explanation

Your program expects this file:

`Dataset/EV_Battery_Dataset.csv`

CSV means Comma-Separated Values. It is a text-based file that stores data in rows and columns.

Imagine a small example dataset:

| Vehicle_Age_Months | Total_Charging_Cycles | Avg_Temperature_C | Fast_Charge_Ratio | Avg_Discharge_Rate_C | SoH_Percent | Battery_Status   |
| ------------------ | --------------------- | ----------------- | ----------------- | -------------------- | ----------- | ---------------- |
| 12                 | 100                   | 25                | 0.10              | 0.5                  | Example     | Healthy          |
| 24                 | 500                   | 30                | 0.25              | 1.5                  | Example     | Healthy          |
| 48                 | 1000                  | 35                | 0.60              | 2.0                  | Example     | Replace Required |

Illustrative table only. The SoH entries are intentionally shown as “Example”; these are not claimed to be real dataset values.

## What is a row?

A row represents one battery record in the dataset.

If your CSV has 1,000 rows, it contains 1,000 records before cleaning, assuming the file has a header row and each remaining row is one record.

## What is a column?

A column stores one type of information.

For example:

- `Vehicle_Age_Months` stores vehicle age.
- `Avg_Temperature_C` stores average temperature.
- `SoH_Percent` stores the known State of Health.
- `Battery_Status` stores the known battery-status label.

## Features and targets

A feature is information given to the model.

A target is the answer the model is trained to predict.

Your project uses five input features:

1. `Vehicle_Age_Months`
2. `Total_Charging_Cycles`
3. `Avg_Temperature_C`
4. `Fast_Charge_Ratio`
5. `Avg_Discharge_Rate_C`

It has two target variables:

- `SoH_Percent` — the regression target.
- `Battery_Status` — the classification target.

The actual number of rows, column names, and values must be verified from your CSV. They cannot be determined from the Python code alone.

# 5. Complete project workflow

Your simplified Python program follows this general order.

### 1

Data collection

Load the CSV file

### 2

Data inspection

Understand the dataset

### 3

Data cleaning

Prepare usable records

### 4

EDA

Study data using charts

### 5

Feature selection

Choose inputs and targets

### 6

Train-test split

Separate training and testing data

### 7

Model training

Teach the models using examples

### 8

Model evaluation

Measure prediction performance

### 9

Cross-validation

Check performance across data splits

### 10

Sample prediction

Predict for a test record

### 11

User input

Predict for new measurements

### 12

Save results

Store reports and cleaned data

The exact code contains 11 numbered project stages in its comments, with data inspection and cleaning included in the early stages. The workflow above separates those operations to make them easier to understand.

# 6. Data collection

The code begins by loading the CSV file.

```
os.makedirs("Reports", exist_ok=True)df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
```

### Line-by-line explanation

Line 1

```
os.makedirs("Reports", exist_ok=True)
```

- Creates a folder named `Reports` if it does not already exist.
- `exist_ok=True` means Python will not raise an error just because the folder already exists.
- `os` is a built-in Python module, not a seventh external library.

Line 2

```
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
```

- `pd` is the usual short name for pandas.
- `read_csv()` opens the CSV file and reads its data.
- The resulting table is stored in `df`.

If the file is missing, the program raises a file-not-found error.

Input: The CSV file.

Output: A pandas DataFrame containing the dataset.

# 7. Data inspection

Before training a model, you need to understand your data.

Your code uses these statements:

```
print(df.head())print("Rows and columns:", df.shape)print("Column names:\n", df.columns.tolist())print("Missing values:\n", df.isnull().sum())print("Duplicate rows:", df.duplicated().sum())print(df.dtypes)print(df.describe(include="all"))print(df.nunique())
```

| Code                         | What it tells you                             |
| ---------------------------- | --------------------------------------------- |
| `df.head()`                  | Shows the first five rows by default.         |
| `df.shape`                   | Returns the number of rows and columns.       |
| `df.columns.tolist()`        | Lists the column names.                       |
| `df.isnull().sum()`          | Counts missing values in each column.         |
| `df.duplicated().sum()`      | Counts rows that duplicate earlier rows.      |
| `df.dtypes`                  | Shows the data type of each column.           |
| `df.describe(include="all")` | Summarizes numerical and categorical columns. |
| `df.nunique()`               | Counts distinct values in each column.        |

## Example: understanding `df.shape`

Suppose this command prints:

```
Rows and columns: (1000, 7)
```

That means there are 1,000 rows and seven columns in that hypothetical dataset.

It does not mean your actual CSV has those dimensions. Run your code to find the real numbers.

## Why is inspection important?

Inspection can reveal:

- Missing battery measurements.
- Repeated records.
- Incorrect column names.
- Text stored where numbers are expected.
- Unusual values.
- An imbalance between Healthy and Replace Required records.

You should inspect these issues before trusting the model.

# 8. Data cleaning

Data cleaning means preparing the dataset so that the model can use it.

Your simplified code is:

```
for col in features + [target_reg]:    df[col] = pd.to_numeric(df[col], errors="coerce")df[target_class] = df[target_class].astype("string").str.strip()df = df.drop_duplicates()df = df.dropna(subset=[target_reg, target_class])
```

## 8.1 Convert numerical columns

```
pd.to_numeric(df[col], errors="coerce")
```

This attempts to convert each feature and the SoH target into numbers.

For example, the text `"25"` can become the number `25`.

If a value cannot be converted, `errors="coerce"` changes it to a missing value (`NaN`).

This makes it possible to identify and handle invalid numerical data.

## 8.2 Clean the status text

```
df[target_class] = df[target_class].astype("string").str.strip()
```

This converts the status column to a string type and removes extra spaces at the beginning and end.

For example:

`" Healthy "` becomes `"Healthy"`.

This helps avoid problems caused by accidental spaces.

## 8.3 Remove duplicate rows

```
df = df.drop_duplicates()
```

`drop_duplicates()` removes repeated rows, keeping the first occurrence by default.

For example:

| Age | Cycles |
| --- | ------ |
| 12  | 100    |
| 12  | 100    |
| 24  | 500    |

The second identical row is removed.

Only fully duplicated rows are removed by this statement. Two records with the same age but different measurements are not duplicates.

## 8.4 Remove rows with missing targets

```
df = df.dropna(subset=[target_reg, target_class])
```

This removes records with missing values in either of the two target columns:

- `SoH_Percent`
- `Battery_Status`

Why? A supervised machine learning model needs a known target answer for training.

Notice that the code does not remove every row containing a missing value. It only removes rows missing a target.

## 8.5 Handle missing feature values

Your models use this pipeline component:

```
SimpleImputer(strategy="median")
```

It fills missing feature values using the median calculated from the training data.

For example, if the training values are 10, 20, and 30, their median is 20. A missing value can be filled with 20.

The actual median depends on the training data.

This imputation step is fitted as part of the model pipeline, so the median is learned from training data rather than from the test set.

## 8.6 Handle outliers

An outlier is a value that is unusually far from other values.

Your code uses the interquartile range (IQR) rule:

```
q1 = df[col].quantile(0.25)q3 = df[col].quantile(0.75)iqr = q3 - q1low = q1 - 1.5 * iqrhigh = q3 + 1.5 * iqrcount = ((df[col] < low) | (df[col] > high)).sum()
```

- `q1` is the 25th percentile.
- `q3` is the 75th percentile.
- `iqr` is the difference between them.
- `low` and `high` are the lower and upper limits used for counting unusual values.

Important: Your simplified code counts outliers but does not remove them. This is deliberate because unusual battery measurements may be real.

# 9. Exploratory Data Analysis (EDA)

EDA stands for Exploratory Data Analysis. It means examining the dataset before interpreting or building a model.

Your project uses printed summaries and several charts.

## 9.1 Sweetviz report

```
report = sv.analyze(df)report.show_html(    "Reports/Sweetviz_EDA_Report.html",    open_browser=False)
```

This generates an HTML report with automated summaries of the dataset.

It can help you inspect distributions, missing values, and relationships.

The report is saved in the `Reports` folder. Your code calls `sv.analyze(df)` without specifying a target column, so it generates a general EDA report rather than setting a specific prediction target.

## 9.2 State of Health histogram

```
plt.hist(df[target_reg].dropna(), bins=20)plt.xlabel("State of Health (%)")plt.ylabel("Frequency")plt.title("Distribution of Battery State of Health")
```

A histogram groups numerical values into ranges called bins.

It helps answer questions such as:

- Which SoH ranges appear most frequently?
- Are the values concentrated in a narrow range?
- Are there unusually low or high observations?

`dropna()` here excludes missing SoH values from the chart. It does not permanently clean the original DataFrame.

## 9.3 SoH box plot

```
plt.boxplot(df[target_reg].dropna(), vert=False)
```

A box plot summarizes the distribution using the median, quartiles, and whiskers.

It can help identify values that may be unusual, but it does not automatically prove that those values are errors.

## 9.4 Charging cycles versus SoH

```
plt.scatter(    df["Total_Charging_Cycles"],    df[target_reg],    alpha=0.5)
```

A scatter plot draws one point per record.

- The horizontal axis represents charging cycles.
- The vertical axis represents SoH.

This helps you inspect whether higher cycle counts tend to appear alongside lower SoH in your dataset.

A visible relationship does not prove that charging cycles alone caused the change.

## 9.5 Correlation heatmap

```
sns.heatmap(    df[features + [target_reg]].corr(),    annot=True,    cmap="coolwarm")
```

The heatmap displays correlation values between the numerical features and SoH.

Correlation values range from -1 to +1:

- Near +1: a strong positive linear relationship.
- Near -1: a strong negative linear relationship.
- Near 0: little or no linear relationship.

Correlation does not prove causation.

## 9.6 Battery status count chart

```
status_counts = df[target_class].value_counts()plt.bar(status_counts.index.astype(str), status_counts.values)
```

This chart compares the number of records in each battery-status category.

It can show whether the dataset contains many Healthy records but very few Replace Required records.

That matters because an uneven distribution can make classification harder to evaluate.

## 9.7 Battery type chart

```
if "Battery_Type" in df.columns:    df["Battery_Type"].value_counts().plot(kind="bar")
```

The chart is created only if the CSV contains a column named `Battery_Type`.

It compares the number of records associated with each battery type.

## Why are charts saved?

Your program calls `plt.savefig()` for each chart.

For example:

```
plt.savefig("Reports/soh_histogram.png")
```

This stores the chart as an image in the `Reports` folder. The chart can then be included in a college report or presentation.

# 10. Feature selection

Feature selection means choosing which columns the model will use.

Your code defines the five features:

```
features = [    "Vehicle_Age_Months",    "Total_Charging_Cycles",    "Avg_Temperature_C",    "Fast_Charge_Ratio",    "Avg_Discharge_Rate_C"]target_reg = "SoH_Percent"target_class = "Battery_Status"
```

## Explanation of the five input features

[Used Renault Zoe - 2013-2024 Reliability & Common Problems | What Car?](https://images.openai.com/static-rsc-4/u7OTWSNAPIiYZBIo-LdUZlF-YDYusZoU-B8lSmKoBjL2rm28IDyNBZnCFKQ5Ovp_YgP4J2ZpMM3Z7MXvLIpyCPGndU6a05Dh8ici5guQFasa07-QS25uXp3iV2QQruM7W8Yv_UQTnYfYMf_iWVw9oMaCnAJsep7_MZWibu3ah0E?purpose=inline)

1\. `Vehicle_Age_Months`

Unit: months

The age of the vehicle, expressed in months.

Examples: `12`, `24`, `36`.

The model uses this feature to learn whether vehicle age is associated with battery health in the training data.

[Close Up of Electric Vehicle EV Charging Plug Connector Connected to a Rapid Charger Station](https://images.openai.com/static-rsc-4/9-EJrLMqIPYjz5TnUnpWxiEs67OmMITMT9zKRXdc3PwfAytUPNThPg_22j-KfAo5KWIP7-ZGlpsv0jtM_GMHfIayIEJaX_cgsI4OjkKQVeA5G1_oZI_PyQiEbzfSQ7nMQDPefu7gwOy2fa0I3uVgkkbA3w4aCIhmlopV5XMrz2PXLZQaZlFC1WEvYmoLLaqd?purpose=inline)

2\. `Total_Charging_Cycles`

Unit: cycles

The recorded number of battery charging cycles.

Examples: `100`, `500`, `1000`.

The exact definition of a cycle must match the dataset's definition.

[Zo werkt: thermomanagement in hybride en EV - deel 1](https://images.openai.com/static-rsc-4/wsqB7ZiHYlmtfEuluKam_0ojgsn6UWq2NSUZvdZ64hzLKPdR68fnzJYG3xyyfAzc10E4pNXnYQ3pg7fuyNMZCUlzcElUAhcddZ3aMtWsgSGDttBtyGq5lpChFP9qi1IzHXmwC5t5qrOtVL2kAnJPhPfVWnK-o2TopzKAo4JOj5A?purpose=inline)

3\. `Avg_Temperature_C`

Unit: degrees Celsius (°C)

The average battery temperature recorded in the dataset.

Examples: `20`, `25`, `35`.

These are examples, not a guaranteed valid range for every battery.

[How Long Does It Take to Charge an Electric Car?](https://images.openai.com/static-rsc-4/tshA8HOTvgT4ZikD6-lfBJIY1Vuv6U5fEDK_T8nT-mh14Ipn7M_ugqueKpR6QFIIb-4IPBcmCYpSXUzhLUDFTS7dGVwOJ5mIDpWM6xNwm6O6giUpgQuw7DTsAqPOaI2zQ8SCe76Je4XmCThTUCaXxRcynCpEKtF5AJuwiMiOTIg?purpose=inline)

4\. `Fast_Charge_Ratio`

Fraction between 0 and 1, as expected by your input code

It represents the proportion of charging counted as fast charging.

Examples: `0`, `0.25`, `0.50`, `0.75`, `1`.

`0.25` represents 25%. Your code rejects values below 0 or above 1.

[LabVIEW开发新能源汽车动力电池功能测试 - 北京瀚文网星科技有限公司](https://images.openai.com/static-rsc-4/QrUsf-LuMG_HX0MrxsNV2aDpZ1_UgQmn3o83nepnocRjbizBYX5db1464wn5pM6ZXSdU8xnTq3ylTryege9oTbgriEN2Nbr--KcxLMAj-L83pF9EIjrn-icTrXPRu7qGQ1zOdR9S0zP7PrzQTZazkTkmhuj_lqrEUsj9LO_rkpY?purpose=inline)

5\. `Avg_Discharge_Rate_C`

Unit: C-rate

The average rate at which the battery releases energy relative to its capacity.

Examples: `0.5`, `1.0`, `2.0`.

The model uses the measurement as recorded in the dataset. The code does not define a separate allowed range for this feature.

## Explanation of the two targets

### Regression target

```
y = df[target_reg]
```

The target is `SoH_Percent`.

The model learns from known SoH values and tries to predict a numerical value for new records.

### Classification target

```
status_map = {"Healthy": 1, "Replace Required": 0}y_class = df[target_class].map(status_map)
```

The code converts the text categories into numbers:

| Original status  | Numerical label |
| ---------------- | --------------- |
| Healthy          | 1               |
| Replace Required | 0               |

Machine learning models can work with numerical class labels. The mapping allows the classifier to learn which of the two categories a record belongs to.

Any other status label becomes a missing value under this mapping and is excluded from the classification dataset.

# 11. Train-test split

A model must be tested on records that were not used to train it.

Your code uses:

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
```

## What do these terms mean?

- `X`: the five input features.
- `y`: the known SoH target.
- `X_train`: training inputs.
- `X_test`: testing inputs.
- `y_train`: training answers.
- `y_test`: testing answers.

`test_size=0.2` means 20% of the records are assigned to the test set, and approximately 80% are assigned to the training set.

For example, with 1,000 usable records:

| Data portion | Approximate number of records | Purpose              |
| ------------ | ----------------------------- | -------------------- |
| Training set | 800                           | Teach the model      |
| Test set     | 200                           | Evaluate predictions |

These counts are hypothetical.

## What does `random_state=42` do?

It makes the random split reproducible. When the data and settings stay the same, rerunning the split generally gives the same allocation.

Your classification code also uses `train_test_split()`. It uses `stratify=y_class` when there are two classes and each class has at least two records. This helps preserve the class proportions across the split.

# 12. Machine learning models

Your simplified project trains four models.

## 12.1 Simple Linear Regression

Simple Linear Regression uses one input feature to predict one numerical target.

Your code uses charging cycles:

```
simple_model.fit(    X_train[["Total_Charging_Cycles"]],    y_train)
```

The model learns a straight-line relationship between charging cycles and SoH.

A simplified equation is:

\\[ \text{SoH} = m \times \text{cycles} + b \\]

Here:

- \\(m\\) is the slope learned by the model.
- \\(b\\) is the intercept learned by the model.

The model estimates these values from the training data.

Why use it? It provides a simple baseline for comparing models that use more features.

## 12.2 Multiple Linear Regression

Multiple Linear Regression uses all five input features to predict SoH.

```
multiple_model.fit(X_train, y_train)
```

Instead of using only charging cycles, it considers:

- Vehicle age.
- Charging cycles.
- Average temperature.
- Fast-charge ratio.
- Average discharge rate.

The model learns how these inputs relate to the numerical SoH target.

It can perform better or worse than Simple Linear Regression, depending on the data. You must compare their test results to know.

## 12.3 Ridge Regression

Ridge Regression is another method for predicting numerical values.

```
ridge_model = Pipeline([    ("missing_values", SimpleImputer(strategy="median")),    ("model", Ridge(alpha=1.0))])
```

Ridge Regression is similar to Linear Regression but adds a penalty that discourages unnecessarily large model coefficients.

The parameter `alpha=1.0` controls the strength of that penalty.

Why use it? It provides another model for comparison and may help when input features are strongly related to each other.

## 12.4 Logistic Regression

Despite its name, Logistic Regression is used here for classification, not numerical SoH prediction.

```
class_model = Pipeline([    ("missing_values", SimpleImputer(strategy="median")),    ("scaling", StandardScaler()),    ("model", LogisticRegression(        max_iter=1000,        class_weight="balanced"    ))])
```

It predicts one of the two status labels:

- `Healthy`
- `Replace Required`

Your pipeline performs two preparation steps:

1. `SimpleImputer` fills missing input values using training-data medians.
2. `StandardScaler` scales numerical features using statistics learned during training.

Then Logistic Regression learns the classification pattern.

`class_weight="balanced"` tells the classifier to account for differences in class frequencies when calculating its training weights.

This does not guarantee that the model will predict the minority class correctly.

## Quick model comparison

| Model                      | Inputs               | Prediction     |
| -------------------------- | -------------------- | -------------- |
| Simple Linear Regression   | Charging cycles only | SoH            |
| Multiple Linear Regression | All five features    | SoH            |
| Ridge Regression           | All five features    | SoH            |
| Logistic Regression        | All five features    | Battery status |

# 13. Model training: what does `fit()` do?

Training means teaching a model using examples with known answers.

For example:

```
multiple_model.fit(X_train, y_train)
```

- `X_train` contains the input measurements.
- `y_train` contains the corresponding known SoH values.
- `fit()` learns the model's parameters from those examples.

Your code trains the other models in the same general way:

```
simple_model.fit(...)ridge_model.fit(...)class_model.fit(...)
```

The difference is that the classification model uses `yc_train`, the numerical status labels, instead of the SoH target.

## What is a pipeline?

A pipeline connects preparation steps and a model into one sequence.

For example:

```
multiple_model = Pipeline([    ("missing_values", SimpleImputer(strategy="median")),    ("model", LinearRegression())])
```

When you call `fit()`:

1. The imputer learns the medians from the training features.
2. It fills any missing training values.
3. Linear Regression learns from the prepared data.

When you later call `predict()`, the same learned imputer is applied to new inputs before the model predicts.

This helps keep training and prediction preparation consistent.

# 14. Model evaluation

After training, your program compares model predictions with the known answers in the test set.

Evaluation helps answer this question:

How well did the model predict records it did not train on?

Your code calculates different metrics for regression and classification.

## 14.1 MAE — Mean Absolute Error

MAE is the average size of the prediction errors, ignoring whether each error is positive or negative.

\\[ \text{MAE}=\frac{1}{n}\sum\_{i=1}^{n}|y_i-\hat y_i| \\]

If the target is SoH measured in percentage points, MAE is measured in percentage points too.

A lower MAE generally indicates smaller prediction errors.

## 14.2 MSE — Mean Squared Error

MSE squares each prediction error before taking the average.

\\[ \text{MSE}=\frac{1}{n}\sum\_{i=1}^{n}(y_i-\hat y_i)^2 \\]

Large errors receive a much larger penalty because they are squared.

MSE is expressed in squared target units.

## 14.3 RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

\\[ \text{RMSE}=\sqrt{\text{MSE}} \\]

It returns the error measurement to the same unit as the target.

A lower RMSE generally means smaller errors, while large errors still receive extra weight.

## 14.4 R² — Coefficient of Determination

R² compares the model's errors with the variation in the target values.

A value closer to 1 generally indicates a better fit on the evaluated dataset.

R² can be negative on test data when a model performs worse than a simple baseline that predicts the test-target mean.

R² is not a percentage-accuracy score.

## 14.5 Accuracy

Accuracy is the proportion of classification predictions that are correct.

\\[ \text{Accuracy}= \frac{\text{Correct predictions}}{\text{All predictions}} \\]

For example, if a hypothetical model correctly predicts 90 out of 100 records, its accuracy is 90%.

But accuracy alone can be misleading if one status category is much more common than the other.

## 14.6 Precision

Precision answers:

When the model predicts a particular class, how often is that prediction correct?

Your code calculates precision for `Replace Required`, which is represented by label `0`.

\\[ \text{Precision}=\frac{TP}{TP+FP} \\]

Here, `TP` and `FP` are defined relative to the class being evaluated. For `Replace Required`, a true positive is a correctly predicted Replace Required record.

## 14.7 Recall

Recall answers:

Of all the records that actually belong to the class, how many did the model identify?

\\[ \text{Recall}=\frac{TP}{TP+FN} \\]

For your replacement class, recall is especially useful for checking how many actual replacement-required records the model detects.

## 14.8 F1-score

F1-score combines precision and recall using their harmonic mean.

\\[ F1=2\times\frac{\text{Precision}\times\text{Recall}} {\text{Precision}+\text{Recall}} \\]

A high F1-score requires both precision and recall to be reasonably high.

Your code uses `zero_division=0` so a zero denominator does not cause an exception in these metric calculations.

## 14.9 Confusion matrix

Your code creates:

```
matrix = confusion_matrix(    yc_test, class_prediction, labels=[0, 1])
```

It compares the actual labels with the predicted labels.

The matrix uses this order:

- Row and column 1: `Replace Required`
- Row and column 2: `Healthy`

The conceptual layout is:

| Actual / Predicted | Replace Required                                    | Healthy                     |
| ------------------ | --------------------------------------------------- | --------------------------- |
| Replace Required   | Correct replacement predictions                     | Missed replacement cases    |
| Healthy            | Healthy records incorrectly flagged for replacement | Correct Healthy predictions |

The cells contain counts, not percentages.

Your program also plots the confusion matrix using a Seaborn heatmap and saves it in the `Reports` folder.

## Which metric is most important?

There is no single best metric for every situation.

For a college comparison:

- MAE and RMSE show the size of SoH prediction errors.
- R² shows how the regression model fits the test data.
- Accuracy gives an overall classification score.
- Precision, recall, and F1-score help explain classification performance for the `Replace Required` class.
- The confusion matrix shows the actual pattern of correct and incorrect classifications.

You must run the program on your dataset to obtain the real metric values.

# 15. Cross-validation

Cross-validation is another way to evaluate a model using multiple data splits.

Your simplified code includes:

```
folds = min(5, len(X))if folds >= 2:    cv_scores = cross_val_score(        Pipeline([            ("missing_values", SimpleImputer(strategy="median")),            ("model", LinearRegression())        ]),        X, y, cv=folds, scoring="r2"    )
```

## How does it work?

Suppose your dataset contains enough records for five folds.

The program divides the dataset into five parts. In each round, it trains the model on four parts and evaluates it on the remaining part.

It repeats this process so that each part is used for testing once.

The program prints the individual R² scores and their mean.

Why use it? One train-test split can give a result that depends on which records were selected. Cross-validation checks performance across several splits.

Important limitation: Your simplified code performs cross-validation for the Linear Regression model used in the regression task. It does not perform classification cross-validation.

# 16. Sample prediction

Your project first makes a prediction for a record already set aside in the test dataset.

```
sample = X_test.iloc[[0]]actual_soh = y_test.iloc[0]predicted_soh = multiple_model.predict(sample)[0]
```

### What does each line mean?

Line 1

```
sample = X_test.iloc[[0]]
```

Selects the first row of the test inputs. The double brackets keep it in a two-dimensional table format, which is the format expected by the model.

Line 2

```
actual_soh = y_test.iloc[0]
```

Gets the known SoH for that same test record.

Line 3

```
predicted_soh = multiple_model.predict(sample)[0]
```

Uses the trained Multiple Linear Regression model to predict SoH for the selected record.

The program then prints the actual and predicted SoH values and saves a bar chart comparing them.

This is different from the later user-input section:

- Sample prediction uses an existing test record with a known answer.
- User-input prediction uses five measurements entered manually.

# 17. User input section — all five measurements

This is one of the most important sections for your college demonstration.

Your code asks for five measurements using `input()` and converts them to floating-point numbers using `float()`.

Here is the same input section from your simplified project:

```
new_battery = pd.DataFrame([{    "Vehicle_Age_Months": float(input("Vehicle age (months): ")),    "Total_Charging_Cycles": float(input("Total charging cycles: ")),    "Avg_Temperature_C": float(input("Average temperature (C): ")),    "Fast_Charge_Ratio": float(input("Fast-charge ratio (0 to 1): ")),    "Avg_Discharge_Rate_C": float(input("Average discharge rate (C-rate): "))}], columns=features)
```

Let's understand every part.

## Input 1: Vehicle age

Prompt:

```
Vehicle age (months):
```

The program expects the age in months.

| Example input | Meaning |
| ------------- | ------- |
| `12`          | 1 year  |
| `24`          | 2 years |
| `36`          | 3 years |
| `60`          | 5 years |

When asked, type a number such as:

```
24
```

Then press Enter.

The model uses the age measurement as one of its five features.

Range limitation: The code does not define a maximum vehicle age. It checks only that age is not negative during the later validation stage.

## Input 2: Total charging cycles

Prompt:

```
Total charging cycles:
```

This is the number of charging cycles recorded for the battery.

| Example input | Meaning               |
| ------------- | --------------------- |
| `100`         | 100 recorded cycles   |
| `500`         | 500 recorded cycles   |
| `800`         | 800 recorded cycles   |
| `1200`        | 1,200 recorded cycles |

When asked, type a number such as:

```
500
```

Then press Enter.

The model uses charging cycles to learn relationships with SoH and battery status.

Range limitation: Your code rejects negative cycle counts, but it does not set a maximum. The dataset's range is the best guide for choosing realistic values.

Also, check how your dataset defines a cycle. A cycle might mean one equivalent full cycle rather than one plug-in session.

## Input 3: Average temperature

Prompt:

```
Average temperature (C):
```

This is the average battery temperature in degrees Celsius.

| Example input | Meaning |
| ------------- | ------- |
| `20`          | 20°C    |
| `25`          | 25°C    |
| `30`          | 30°C    |
| `35`          | 35°C    |

When asked, type a number such as:

```
30
```

Then press Enter.

The model uses this temperature measurement as a feature.

Range limitation: The code does not set a minimum or maximum temperature. Choose a value consistent with the units, measurement conditions, and range of your dataset.

## Input 4: Fast-charge ratio

Prompt:

```
Fast-charge ratio (0 to 1):
```

This represents the proportion of charging that your dataset classifies as fast charging.

| Input  | Equivalent percentage |
| ------ | --------------------- |
| `0`    | 0%                    |
| `0.10` | 10%                   |
| `0.25` | 25%                   |
| `0.50` | 50%                   |
| `0.75` | 75%                   |
| `1`    | 100%                  |

When asked, type:

```
0.25
```

Then press Enter.

Do not type `25` to mean 25%. Your code will reject `25` because it requires a value between 0 and 1.

The validation is:

```
if not 0 <= new_battery.loc[0, "Fast_Charge_Ratio"] <= 1:    raise ValueError("Fast-charge ratio must be between 0 and 1.")
```

This is the only one of the five inputs with a clearly defined upper and lower bound in the code.

The exact meaning of the ratio still depends on how the dataset calculates it.

## Input 5: Average discharge rate

Prompt:

```
Average discharge rate (C-rate):
```

This describes the average discharge rate relative to the battery's capacity.

| Example input | General interpretation                         |
| ------------- | ---------------------------------------------- |
| `0.5`         | Half the nominal one-hour rate                 |
| `1.0`         | Nominal one-hour rate                          |
| `1.5`         | One-and-a-half times the nominal one-hour rate |
| `2.0`         | Twice the nominal one-hour rate                |

These are general interpretations of C-rate, not claims about your specific battery dataset.

When asked, type:

```
1.5
```

Then press Enter.

The model uses this measurement as its fifth input feature.

Range limitation: The code does not define a separate minimum or maximum discharge rate. Use realistic values consistent with your dataset.

## The correct order of terminal inputs

Your program asks for these values in this order:

1\. Vehicle age

Example: `24`

2\. Charging cycles

Example: `500`

3\. Average temperature

Example: `30`

4\. Fast-charge ratio

Example: `0.25`\*\*

5\. Average discharge rate

Example: `1.5`\*\*

These example values are for practice. Before using them in a demonstration, compare them with the actual ranges in your CSV dataset.

# 18. Prediction calculation — what happens inside the code?

After receiving the five values, the program creates a DataFrame and passes it to the trained models.

## Step 1: Convert the inputs to numbers

For example:

```
float(input("Vehicle age (months): "))
```

There are two functions here.

- `input()` displays a prompt and receives the user's response as text.
- `float()` converts that text to a decimal number.

For example, if you type `"24"`, `float()` converts it to the numerical value `24.0`.

If you type `"twenty-four"`, conversion fails because that text is not a valid number.

## Step 2: Create a Pandas DataFrame

Your code puts the five values into a one-row table:

```
new_battery = pd.DataFrame([{    "Vehicle_Age_Months": 24,    "Total_Charging_Cycles": 500,    "Avg_Temperature_C": 30,    "Fast_Charge_Ratio": 0.25,    "Avg_Discharge_Rate_C": 1.5}], columns=features)
```

The numbers above are illustrative.

The `columns=features` argument specifies the expected column order. This is important because the trained models expect the same feature names and order used during training.

## Step 3: Predict State of Health

Your code uses:

```
soh = float(np.clip(    multiple_model.predict(new_battery)[0], 0, 100))
```

Here's the order:

1. `multiple_model.predict(new_battery)` sends the new measurements to the trained regression model.
2. `[0]` selects the prediction for the first row.
3. `np.clip(..., 0, 100)` limits the displayed numerical result to 0–100.
4. `float()` converts the result into a standard Python floating-point number.

If the model returns an illustrative prediction of `86.4`, the displayed SoH becomes `86.4%`.

The actual result can only be obtained by running the model with your dataset.

## Step 4: Predict battery status

```
status_number = int(class_model.predict(new_battery)[0])
```

The classifier returns one of the numerical class labels:

- `1` means Healthy.
- `0` means Replace Required.

Your program then converts that number into a readable string:

```
status = "Healthy" if status_number == 1 else "Replace Required"
```

This is a conditional expression: if the predicted number is 1, use `"Healthy"`; otherwise, use `"Replace Required"`.

## Step 5: Get class probabilities

```
probability_values = class_model.predict_proba(new_battery)[0]class_labels = list(class_model.named_steps["model"].classes_)healthy_probability = (    probability_values[class_labels.index(1)] * 100)
```

`predict_proba()` returns the estimated probability for each class.

The code finds the probability belonging to class `1`, which means Healthy, and multiplies it by 100 to express it as a percentage.

For example, a model might return a Healthy probability of 0.82, which is displayed as 82%.

That number is an estimated class probability, not a guarantee that the battery is healthy and not the same as the model's overall test accuracy.

## Step 6: Generate advice

Your code uses:

```
if status == "Replace Required":    advice = "Arrange a professional battery inspection."elif soh < 80:    advice = "Arrange a battery health check soon."else:    advice = "Continue routine monitoring."
```

The order is important:

1. If the predicted status is Replace Required, the first condition is selected.
2. Otherwise, if SoH is below 80, the second condition is selected.
3. Otherwise, the third condition is selected.

The threshold of 80% is a rule in your code. It is not a universal replacement standard.

# 19. Output explanation

After prediction, the code prints the status, SoH, Healthy probability, and advice.

```
print("\nPredicted battery status:", status)print("Predicted SoH:", round(soh, 2), "%")print("Estimated Healthy probability:", round(healthy_probability, 2), "%")print("Advice:", advice)
```

Here's what each result means.

| Output                        | Meaning                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| Predicted battery status      | The category chosen by Logistic Regression.                                           |
| Predicted SoH                 | The numerical estimate from Multiple Linear Regression, limited to 0–100 for display. |
| Estimated Healthy probability | The classifier's estimated probability for the Healthy class.                         |
| Advice                        | A message selected by the program's conditional rules.                                |

`round(soh, 2)` rounds the displayed number to two decimal places.

For example, `86.437` would display as `86.44`.

The program also creates a chart:

```
plt.figure(figsize=(7, 5))plt.bar(["Predicted SoH"], [soh])plt.ylim(0, 105)plt.ylabel("State of Health (%)")plt.title("New Battery Prediction")plt.savefig("Reports/deployment_prediction.png")plt.show()plt.close()
```

This chart shows the predicted SoH as a bar and saves the image to `Reports/deployment_prediction.png`.

# 20. Deployment — how to run the project on your computer

In this project, deployment means running the trained workflow locally and allowing a user to enter new battery measurements through the terminal.

It is not yet a website or mobile app.

## 20.1 Prepare the folder structure

Your files should be arranged like this:

EV_Battery_Project/

`ev_battery_project_simple.py`

`Dataset/`

`EV_Battery_Dataset.csv`

`Reports/` — created automatically

The script expects to be run from the main folder so that the relative dataset path works.

## 20.2 Install the libraries

Open the terminal and run:

```
pip install pandas numpy matplotlib seaborn scikit-learn sweetviz
```

If your computer uses a Python command named `python3`, use that command to run the program instead.

## 20.3 Run the program

```
python ev_battery_project_simple.py
```

The program executes its steps in order:

1. Loads the dataset.
2. Prints the inspection results.
3. Cleans the data.
4. Creates EDA reports and charts.
5. Selects features and targets.
6. Splits the dataset.
7. Trains the models.
8. Evaluates model performance.
9. Runs regression cross-validation.
10. Makes a sample prediction.
11. Requests new battery inputs.
12. Saves the results.

Some charts use `plt.show()`. If a chart window opens and pauses the program, close the window to continue.

## 20.4 What is the Reports folder for?

The program saves its outputs in `Reports`.

Examples include:

- `Sweetviz_EDA_Report.html`
- `soh_histogram.png`
- `soh_boxplot.png`
- `cycles_vs_soh.png`
- `correlation_heatmap.png`
- `battery_status_counts.png`
- `battery_type_bar.png`, if the dataset has `Battery_Type`
- `simple_regression_line.png`
- `confusion_matrix.png`
- `sample_prediction.png`
- `deployment_prediction.png`
- `regression_results.csv`
- `classification_results.csv`
- `cleaned_ev_battery_dataset.csv`

Not every optional file is guaranteed to be created. For example, the battery-type chart is conditional on the column existing, and the prediction chart requires a successful prediction.

The cleaned dataset is saved after the prediction section. If an error stops the program earlier, the later output files may not be written.

## 20.5 Complete hypothetical terminal interaction

Below is an example of the input section's format.

The output values are illustrative only. They are not actual predictions or metrics from your dataset.

```
Enter new battery measurements.

Vehicle age (months): 24
Total charging cycles: 500
Average temperature (C): 30
Fast-charge ratio (0 to 1): 0.25
Average discharge rate (C-rate): 1.5

Predicted battery status: Healthy
Predicted SoH: 86.40 %
Estimated Healthy probability: 82.00 %
Advice: Continue routine monitoring.
Educational prediction only; not a certified battery safety test.
```

Your real output may differ. You should never write these example results in your report as though your model actually produced them.

# 21. More practical examples

You can test your program with several different sets of inputs.

The following examples are hypothetical. They show what you can type, not what the model will predict.

## Choose a practice test

Example A — lower age and cycle count

Example B — middle-range practice values

Example C — higher age and cycle count

Example D — another combination for comparison

### Values to enter

| Vehicle age            | 12 months  |
| ---------------------- | ---------- |
| Charging cycles        | 100        |
| Average temperature    | 25 °C      |
| Fast-charge ratio      | 0.1        |
| Average discharge rate | 0.5 C-rate |

&#x20; Copy example inputs&#x20;

These values must be checked against your CSV ranges before you use them for a formal demonstration.

## How to compare the tests

Run the program separately for each example.

Write down the outputs produced by your own model:

| Test | Predicted SoH        | Predicted status     | Healthy probability  |
| ---- | -------------------- | -------------------- | -------------------- |
| A    | Record actual output | Record actual output | Record actual output |
| B    | Record actual output | Record actual output | Record actual output |
| C    | Record actual output | Record actual output | Record actual output |
| D    | Record actual output | Record actual output | Record actual output |

You can then compare the results and explain how changing the inputs affects the model's predictions.

Do not assume that a higher cycle count must always produce a particular prediction. The model's learned relationships and all five features influence the output. Only actual runs can establish what your trained model predicts.

# 22. Common errors and solutions

| Error                                | Why it happens                                                                        | Solution                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `FileNotFoundError`                  | The CSV file cannot be found at the specified path.                                   | Put `EV_Battery_Dataset.csv` inside `Dataset` and run the script from the main project folder. |
| `ModuleNotFoundError`                | A required library is missing from the Python environment.                            | Run the provided installation command in the same environment.                                 |
| `could not convert string to float`  | The user enters text instead of a number.                                             | Type `24`, not `twenty-four`.                                                                  |
| Fast-charge ratio error              | The ratio is below 0 or above 1.                                                      | Enter values such as `0.25`, not `25` for 25%.                                                 |
| Unexpected battery-status labels     | The dataset has labels that are not in the mapping.                                   | Check the CSV labels. The code expects `Healthy` and `Replace Required` for classification.    |
| Missing target values                | Some records have no SoH or status label.                                             | The cleaning stage removes records missing either target.                                      |
| Too few records for evaluation       | There are not enough usable rows or class examples.                                   | Check the dataset size and the number of records in each class.                                |
| Incorrect column name                | The code refers to a column that does not exist.                                      | Check `df.columns` and make sure the CSV headers match the code.                               |
| A chart appears to stop the program  | `plt.show()` is waiting for the chart window to close.                                | Close the chart window to continue.                                                            |
| The program skips the new prediction | Input conversion or validation raises a `ValueError`, or the input ends unexpectedly. | Enter valid numerical inputs and keep the fast-charge ratio between 0 and 1.                   |

## A limitation of the current input validation

Your code explicitly checks that:

- Vehicle age is not negative.
- Charging cycles are not negative.
- Fast-charge ratio is between 0 and 1.

It does not independently validate every other input against a dataset-specific range. A numerical temperature or discharge rate can be accepted by the code even if it is unrealistic.

The `try/except` block catches `ValueError` and `EOFError` during the interactive input section. In those cases, it prints a message and skips the new-battery prediction.

# 23. Project limitations

It is important to understand what your project can and cannot do.

1\. The model depends on the dataset. If the training data is small, inaccurate, or unrepresentative, predictions may be unreliable.

2\. Classification may be affected by class imbalance. If there are very few Replace Required records, the model may not learn that class well.

3\. The prediction is not a physical battery test. The model uses the five measurements you enter; it does not directly inspect the battery.

4\. The model may encounter unfamiliar values. A prediction can still be generated for a value outside the training range, but its reliability is uncertain.

5\. SoH clipping does not fix model errors. `np.clip()` only limits the numerical result to 0–100. It does not make the model more accurate.

6\. The Healthy probability is not guaranteed confidence. It is a probability estimated by the classifier and may not match the model's real-world reliability.

7\. The 80% threshold is a code rule. It should not be treated as a universal battery replacement recommendation.

8\. No live sensor connection exists in this project. The program receives measurements from the user through the terminal.

9\. Results need validation. Actual model performance should be checked with the real dataset, suitable test data, and, for real applications, independent technical validation.

# 24. Viva preparation

Here are common questions a teacher might ask, with short answers you can learn.

1\. What is your project?My project uses Python and machine learning to predict EV battery State of Health and battery status.

2\. What is SoH?

3\. Why did you choose this project?

4\. What is machine learning?

5\. Which libraries do you use?

6\. What is a dataset?

7\. What is a feature?

8\. What are your five features?

9\. What are your targets?

10\. What is regression?

11\. What is classification?

12\. Why use Linear Regression?

13\. Why use Logistic Regression?

14\. What does fit() do?

15\. What does predict() do?

16\. What is train-test split?

17\. Why use an 80:20 split?

18\. What is MAE?

19\. What is R²?

20\. What is a confusion matrix?

21\. What is cross-validation?

22\. Why use drop_duplicates()?

23\. Why use dropna()?

24\. What is predict_proba()?

25\. What is np.clip()?

26\. What does Fast_Charge_Ratio mean?

27\. Why do you use a DataFrame for new inputs?

28\. What is deployment in your project?

29\. Does your project connect to a real battery sensor?

30\. Can the prediction guarantee battery safety?

## A short introduction for your viva

You can practise saying this aloud:

Good morning, sir or madam.

My project is EV Battery Health Prediction using Python and Machine Learning. The purpose of this project is to estimate the State of Health of an electric vehicle battery and predict its status.

I use a CSV dataset containing five input features: vehicle age, total charging cycles, average temperature, fast-charge ratio, and average discharge rate.

First, I load and inspect the dataset. Then I clean the data, perform exploratory data analysis, select the features, and split the dataset into training and testing sets.

I train Simple Linear Regression, Multiple Linear Regression, Ridge Regression, and Logistic Regression models. I evaluate their performance using suitable metrics.

Finally, the program accepts five new measurements from the user and predicts SoH and battery status. The results are educational estimates, not certified battery diagnostics.

Thank you.

# 25. Final project summary

Your EV Battery Health Prediction project uses Python and machine learning to estimate battery State of Health and predict battery status from five input measurements.

First, the program loads the CSV dataset using pandas. It inspects the data, removes duplicate records, handles missing target values, and prepares missing input features through scikit-learn pipelines. Next, it uses charts and a Sweetviz report to explore the dataset.

The program then selects the five features, divides the data into training and testing sets, and trains four machine learning models. It evaluates regression performance using MAE, MSE, RMSE, and R², and classification performance using accuracy, precision, recall, F1-score, and a confusion matrix. Cross-validation provides an additional evaluation of the regression model.

Finally, the user enters five new battery measurements through the terminal. The trained regression model predicts SoH, and the classification model predicts `Healthy` or `Replace Required`. The program displays the results, generates a prediction chart, and saves reports and result files in the `Reports` folder.

The project demonstrates the complete machine learning process, from data collection to prediction. Its results are educational estimates and must not be treated as certified battery safety assessments.

## Quick revision: remember these five points

1\. Data

Load, inspect, and clean the EV battery dataset.

2\. Analysis

Study the measurements using EDA charts and reports.

3\. Training

Train regression and classification models on known examples.

4\. Evaluation

Measure how well the models predict records they did not train on.

5\. Prediction

Enter five battery measurements and obtain an estimated SoH and battery-status category.

The most important distinction for your presentation: Multiple Linear Regression predicts the numerical SoH, while Logistic Regression predicts the battery-status category. Both models use the five input features, but they solve different prediction tasks.
