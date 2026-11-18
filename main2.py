# EV BATTERY HEALTH PREDICTION PROJECT
# Uses only: pandas, numpy, matplotlib, seaborn, scikit-learn, sweetviz

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
# FIX: one folder name used everywhere, so saving files never fails
REPORT_FOLDER = "Main/Main-2-Reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")

print("\nDATASET")
print(df.head())
print("\nRows and columns:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:\n")
print(df.dtypes)
print("\nStatistics:\n", df.describe(include="all"))
print("\nUnique values:\n", df.nunique())

# ---------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------
features = [
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C"
]
target_reg = "SoH_Percent"
target_class = "Battery_Status"
required = features + [target_reg, target_class]

missing_columns = [col for col in required if col not in df.columns]
if missing_columns:
    raise ValueError("Missing dataset columns: " + str(missing_columns))

for col in features + [target_reg]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target_class] = df[target_class].astype("string").str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=[target_reg, target_class])

# The teacher's sample removes target outliers using IQR.
# For battery features, outliers are counted and kept because they may be valid.
print("\nMissing values after cleaning:\n", df[required].isnull().sum())
print("Rows after cleaning:", len(df))

for col in features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = ((df[col] < low) | (df[col] > high)).sum()
    print("Outliers in", col, ":", int(count))

# Save a simple dataset summary table.
summary = pd.DataFrame({
    "Check": ["Rows", "Columns", "Missing cells", "Duplicate rows"],
    "Value": [
        len(df), df.shape[1],
        int(df[required].isnull().sum().sum()),
        int(df.duplicated().sum())
    ]
})
print("\nDataset summary:\n", summary.to_string(index=False))

# ---------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
print("\nSoH summary:\n", df[target_reg].describe())
print("\nBattery status counts:\n", df[target_class].value_counts())

try:
    report = sv.analyze(df)
    report.show_html(f"{REPORT_FOLDER}/Sweetviz_EDA_Report.html", open_browser=False)
    print("Sweetviz report saved in", REPORT_FOLDER)
except Exception as error:
    print("Sweetviz report skipped:", error)

plt.figure(figsize=(8, 5))
plt.hist(df[target_reg].dropna(), bins=20)
plt.xlabel("State of Health (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Battery State of Health")
plt.savefig(f"{REPORT_FOLDER}/soh_histogram.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot(df[target_reg].dropna(), vert=False)
plt.xlabel("State of Health (%)")
plt.title("Battery State of Health Box Plot")
plt.savefig(f"{REPORT_FOLDER}/soh_boxplot.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["Total_Charging_Cycles"], df[target_reg], alpha=0.5)
plt.xlabel("Total Charging Cycles")
plt.ylabel("State of Health (%)")
plt.title("Charging Cycles vs Battery Health")
plt.savefig(f"{REPORT_FOLDER}/cycles_vs_soh.png")
plt.show()
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[features + [target_reg]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig(f"{REPORT_FOLDER}/correlation_heatmap.png")
plt.show()
plt.close()

status_counts = df[target_class].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(status_counts.index.astype(str), status_counts.values)
plt.xlabel("Battery Status")
plt.ylabel("Number of Records")
plt.title("Battery Status Counts")
plt.savefig(f"{REPORT_FOLDER}/battery_status_counts.png")
plt.show()
plt.close()

if "Battery_Type" in df.columns:
    plt.figure(figsize=(7, 5))
    df["Battery_Type"].value_counts().plot(kind="bar")
    plt.xlabel("Battery Type")
    plt.ylabel("Number of Records")
    plt.title("Records by Battery Type")
    plt.savefig(f"{REPORT_FOLDER}/battery_type_bar.png")
    plt.show()
    plt.close()

# ---------------------------------------------------------
# 4. Feature selection and train-test split
# ---------------------------------------------------------
X = df[features]
y = df[target_reg]

# The status model uses Battery_Status, not SoH_Percent, as its target.
status_map = {"Healthy": 1, "Replace Required": 0}
y_class = df[target_class].map(status_map)

valid_class_rows = y_class.notna()
X_class = X.loc[valid_class_rows]
y_class = y_class.loc[valid_class_rows].astype(int)

print("\nInput features:", features)
print("Regression target:", target_reg)
print("Classification target:", target_class)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratify only when both classes have enough records.
if len(y_class) >= 5 and y_class.value_counts().min() >= 2:
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
else:
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

print("Regression training rows:", len(X_train))
print("Regression testing rows:", len(X_test))

# ---------------------------------------------------------
# 5. Build models
# ---------------------------------------------------------
simple_model = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])
multiple_model = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])
ridge_model = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("model", Ridge(alpha=1.0))
])
class_model = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("scaling", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

simple_model.fit(X_train[["Total_Charging_Cycles"]], y_train)
multiple_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
class_model.fit(Xc_train, yc_train)

print("\nModels trained successfully.")
print("Simple regression equation: SoH =",
      round(simple_model.named_steps["model"].coef_[0], 4),
      "* charging cycles +",
      round(simple_model.named_steps["model"].intercept_, 4))

# Plot regression line
ordered = df.sort_values("Total_Charging_Cycles")
plt.figure(figsize=(8, 5))
plt.scatter(df["Total_Charging_Cycles"], df[target_reg], alpha=0.4)
plt.plot(
    ordered["Total_Charging_Cycles"],
    simple_model.predict(ordered[["Total_Charging_Cycles"]]),
    linewidth=2
)
plt.xlabel("Total Charging Cycles")
plt.ylabel("State of Health (%)")
plt.title("Simple Linear Regression")
plt.savefig(f"{REPORT_FOLDER}/simple_regression_line.png")
plt.show()
plt.close()

# ---------------------------------------------------------
# 6. Evaluate regression models
# ---------------------------------------------------------
results = []
for name, model, test_x in [
    ("Simple Linear Regression", simple_model, X_test[["Total_Charging_Cycles"]]),
    ("Multiple Linear Regression", multiple_model, X_test),
    ("Ridge Regression", ridge_model, X_test)
]:
    prediction = model.predict(test_x)
    mse = mean_squared_error(y_test, prediction)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, prediction)
    })

results_df = pd.DataFrame(results)
print("\nRegression results:\n", results_df.round(4).to_string(index=False))

# ---------------------------------------------------------
# 7. Evaluate classification model
# ---------------------------------------------------------
class_prediction = class_model.predict(Xc_test)
class_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Replace Required)",
               "Recall (Replace Required)", "F1 (Replace Required)"],
    "Score": [
        accuracy_score(yc_test, class_prediction),
        precision_score(yc_test, class_prediction, pos_label=0, zero_division=0),
        recall_score(yc_test, class_prediction, pos_label=0, zero_division=0),
        f1_score(yc_test, class_prediction, pos_label=0, zero_division=0)
    ]
})
print("\nClassification results:\n", class_results.round(4).to_string(index=False))

matrix = confusion_matrix(yc_test, class_prediction, labels=[0, 1])
print("\nConfusion matrix (Replace Required, Healthy):\n", matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Replace Required", "Healthy"],
            yticklabels=["Replace Required", "Healthy"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{REPORT_FOLDER}/confusion_matrix.png")
plt.show()
plt.close()

# ---------------------------------------------------------
# 8. Cross-validation
# ---------------------------------------------------------
folds = min(5, len(X))
if folds >= 2:
    cv_scores = cross_val_score(
        Pipeline([
            ("missing_values", SimpleImputer(strategy="median")),
            ("model", LinearRegression())
        ]),
        X, y, cv=folds, scoring="r2"
    )
    print("\nCross-validation R2 scores:", np.round(cv_scores, 4))
    print("Mean cross-validation R2:", round(cv_scores.mean(), 4))

# ---------------------------------------------------------
# 9. Test one battery record
# ---------------------------------------------------------
sample = X_test.iloc[[0]]
actual_soh = y_test.iloc[0]
predicted_soh = multiple_model.predict(sample)[0]
print("\nSample actual SoH:", round(actual_soh, 2))
print("Sample predicted SoH:", round(predicted_soh, 2))

plt.figure(figsize=(7, 5))
plt.bar(["Actual SoH", "Predicted SoH"], [actual_soh, predicted_soh])
plt.ylabel("State of Health (%)")
plt.title("Actual vs Predicted SoH")
plt.savefig(f"{REPORT_FOLDER}/sample_prediction.png")
plt.show()
plt.close()

# ---------------------------------------------------------
# 10. Predict a new battery from user input
# ---------------------------------------------------------
print("\nEnter new battery measurements (numbers only).")
try:
    new_battery = pd.DataFrame([{
        "Vehicle_Age_Months": float(input("Vehicle age (months): ")),
        "Total_Charging_Cycles": float(input("Total charging cycles: ")),
        "Avg_Temperature_C": float(input("Average temperature (C): ")),
        "Fast_Charge_Ratio": float(input("Fast-charge ratio (0 to 1): ")),
        "Avg_Discharge_Rate_C": float(input("Average discharge rate (C-rate): "))
    }], columns=features)

    if not 0 <= new_battery.loc[0, "Fast_Charge_Ratio"] <= 1:
        raise ValueError("Fast-charge ratio must be between 0 and 1.")

    soh = float(np.clip(multiple_model.predict(new_battery)[0], 0, 100))
    status_number = int(class_model.predict(new_battery)[0])
    probability_values = class_model.predict_proba(new_battery)[0]
    class_labels = list(class_model.named_steps["model"].classes_)
    healthy_probability = probability_values[class_labels.index(1)] * 100

    status = "Healthy" if status_number == 1 else "Replace Required"
    if status == "Replace Required":
        advice = "Arrange a professional battery inspection."
    elif soh < 80:
        advice = "Arrange a battery health check soon."
    else:
        advice = "Continue routine monitoring."

    print("\nPredicted battery status:", status)
    print("Predicted SoH:", round(soh, 2), "%")
    print("Estimated Healthy probability:", round(healthy_probability, 2), "%")
    print("Advice:", advice)
    print("Educational prediction only; not a certified battery safety test.")

    plt.figure(figsize=(7, 5))
    plt.bar(["Predicted SoH"], [soh])
    plt.ylim(0, 105)
    plt.ylabel("State of Health (%)")
    plt.title("New Battery Prediction")
    plt.savefig(f"{REPORT_FOLDER}/deployment_prediction.png")
    plt.show()
    plt.close()

except (ValueError, EOFError) as error:
    print("New battery prediction skipped:", error)

# ---------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------
results_df.to_csv(f"{REPORT_FOLDER}/regression_results.csv", index=False)
class_results.to_csv(f"{REPORT_FOLDER}/classification_results.csv", index=False)
df.to_csv(f"{REPORT_FOLDER}/cleaned_ev_battery_dataset.csv", index=False)

print("\nProject completed.")
print("Charts and results are saved in", REPORT_FOLDER)