import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# FIX: one folder name used everywhere, so saving charts never fails
REPORT_FOLDER = "Main/Main-Reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load the EV battery dataset
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nBattery status counts:\n", df["Battery_Status"].value_counts())

# Prepare the data
feature_columns = [
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
]
target_column = "SoH_Percent"

for column in feature_columns + [target_column]:
    df[column] = df[column].fillna(df[column].median())
df["Battery_Status"] = df["Battery_Status"].fillna(
    df["Battery_Status"].mode()[0]
)
df = df.drop_duplicates()

# Explore the data
plt.figure(figsize=(8, 5))
plt.hist(df[target_column], bins=20)
plt.xlabel("Battery State of Health (%)")
plt.ylabel("Number of batteries")
plt.title("Battery State of Health")
plt.tight_layout()
plt.savefig(f"{REPORT_FOLDER}/soh_histogram.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(df[target_column])
plt.ylabel("Battery State of Health (%)")
plt.title("State of Health Box Plot")
plt.tight_layout()
plt.savefig(f"{REPORT_FOLDER}/soh_boxplot.png")
plt.show()
plt.close()

status_counts = df["Battery_Status"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(status_counts, labels=status_counts.index, autopct="%1.1f%%")
plt.title("Battery Status")
plt.tight_layout()
plt.savefig(f"{REPORT_FOLDER}/battery_status_pie.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(
    df[feature_columns + [target_column]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Battery Feature Correlation")
plt.tight_layout()
plt.savefig(f"{REPORT_FOLDER}/correlation_heatmap.png")
plt.show()
plt.close()

# Select inputs and targets
X = df[feature_columns]
y_soh = df[target_column]
y_status = (df["Battery_Status"] == "Healthy").astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_soh,
    test_size=0.2,
    random_state=42,
)
X_train_status, X_test_status, y_train_status, y_test_status = train_test_split(
    X,
    y_status,
    test_size=0.2,
    random_state=42,
    stratify=y_status,
)

# Train the SoH prediction models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge_model.fit(X_train, y_train)

# Train the battery status model
status_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
status_model.fit(X_train_status, y_train_status)

# Check model results
soh_predictions = ridge_model.predict(X_test)
linear_predictions = linear_model.predict(X_test)
print("\nLinear Regression Results")
print("MAE:", mean_absolute_error(y_test, linear_predictions))
print("MSE:", mean_squared_error(y_test, linear_predictions))
print("R2:", r2_score(y_test, linear_predictions))

print("\nRidge Regression Results")
print("MAE:", mean_absolute_error(y_test, soh_predictions))
print("MSE:", mean_squared_error(y_test, soh_predictions))
print("R2:", r2_score(y_test, soh_predictions))

cv_scores = cross_val_score(
    LinearRegression(),
    X,
    y_soh,
    cv=5,
    scoring="r2",
)
print("\nLinear Regression Cross-Validation R2 Scores:", cv_scores)
print("Mean R2:", cv_scores.mean())

status_predictions = status_model.predict(X_test_status)
print("\nBattery Status Accuracy:", accuracy_score(y_test_status, status_predictions))
print("Status labels: 1 = Healthy, 0 = Replace Required")
print("Only four rows have the Replace Required label, so status results are limited.")

# Predict using new battery measurements
print("\nEnter the battery measurements (numbers only):")
new_battery = pd.DataFrame(
    [[
        float(input("Vehicle age (months): ")),
        float(input("Total charging cycles: ")),
        float(input("Average temperature (C): ")),
        float(input("Fast-charge ratio (0 to 1): ")),
        float(input("Average discharge rate (C-rate): ")),
    ]],
    columns=feature_columns,
)

soh_prediction = ridge_model.predict(new_battery)[0]
soh_prediction = max(0, min(100, soh_prediction))
status_prediction = status_model.predict(new_battery)[0]

if status_prediction == 1:
    battery_status = "Healthy"
else:
    battery_status = "Replace Required"

print("\nPredicted State of Health:", f"{soh_prediction:.2f}%")
print("Predicted Battery Status:", battery_status)
print("This project is for learning only, not for real battery safety decisions.")