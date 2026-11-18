# EV Battery Health Prediction
# Run from the main project folder with: python Python_Code/project.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,f1_score,mean_absolute_error,mean_squared_error,precision_score,r2_score,recall_score,)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


print("Stage 1: Problem definition")
print("Learn to predict battery SoH (%) and status from the supplied dataset.")
print("This synthetic-data project is not a battery safety test.")

print("\nStage 2: Data collection")
df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")
print("Dataset shape (rows, columns):", df.shape)
print("Column names:", list(df.columns))
print("\nFirst five rows:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nNumeric summary:")
print(df.describe())
print("\nBattery status counts:")
print(df["Battery_Status"].value_counts())

summary = pd.DataFrame(
    {
        "Check": [
            "Rows",
            "Columns",
            "Missing cells",
            "Duplicate rows",
            "Healthy labels",
            "Replace Required labels",
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
            int((df["Battery_Status"] == "Healthy").sum()),
            int((df["Battery_Status"] == "Replace Required").sum()),
        ],
    }
)
plt.figure(figsize=(9, 3))
plt.axis("off")
plt.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Dataset summary")
plt.tight_layout()
plt.savefig("Reports/dataset_summary.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 3: Data preparation")
feature_columns = [
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
]
required_columns = feature_columns + ["SoH_Percent", "Battery_Status"]
rows_before = len(df)
missing_before = int(df[required_columns].isna().sum().sum())
duplicates_before = int(df.duplicated().sum())
outliers_before = 0

for col in feature_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    low = q1 - 1.5 * (q3 - q1)
    high = q3 + 1.5 * (q3 - q1)
    outliers_before += int(((df[col] < low) | (df[col] > high)).sum())

df = df.drop_duplicates()
df = df.dropna(subset=required_columns)
outliers_kept = 0

for col in feature_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    low = q1 - 1.5 * (q3 - q1)
    high = q3 + 1.5 * (q3 - q1)
    outliers_kept += int(((df[col] < low) | (df[col] > high)).sum())

cleaning = pd.DataFrame(
    {
        "Check": [
            "Rows",
            "Missing target or input cells",
            "Duplicate rows",
            "Input values outside IQR limits",
        ],
        "Before": [
            rows_before,
            missing_before,
            duplicates_before,
            outliers_before,
        ],
        "After": [
            len(df),
            int(df[required_columns].isna().sum().sum()),
            int(df.duplicated().sum()),
            outliers_kept,
        ],
    }
)
print("Preparation summary:")
print(cleaning.to_string(index=False))
print("IQR outliers are counted, not removed; they may be valid observations.")

plt.figure(figsize=(10, 3))
plt.axis("off")
plt.table(
    cellText=cleaning.values,
    colLabels=cleaning.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Data preparation checks")
plt.tight_layout()
plt.savefig("Reports/data_cleaning.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 4: Exploratory data analysis")
print("SoH summary (%):")
print(df["SoH_Percent"].describe())
print("Fast-charge ratio range:", df["Fast_Charge_Ratio"].min(), "to", df["Fast_Charge_Ratio"].max())
print("Creating the Sweetviz EDA report...")
eda_df = df.copy()
eda_df["Healthy"] = df["Battery_Status"] == "Healthy"
eda_report = sv.analyze(eda_df, target_feat="Healthy")
eda_report.show_html(
    filepath="Reports/Sweetviz_EDA_Report.html",
    open_browser=False,
)
print("Sweetviz report saved: Reports/Sweetviz_EDA_Report.html")

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="SoH_Percent", bins=30, color="#247ba0")
plt.title("Battery State of Health distribution")
plt.xlabel("State of Health (%)")
plt.ylabel("Number of records (rows)")
plt.tight_layout()
plt.savefig("Reports/soh_histogram.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot(df["SoH_Percent"], orientation="horizontal")
plt.title("Battery State of Health box plot")
plt.xlabel("State of Health (%)")
plt.ylabel("Battery records")
plt.tight_layout()
plt.savefig("Reports/soh_boxplot.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(
    df["Total_Charging_Cycles"],
    df["SoH_Percent"],
    s=12,
    alpha=0.4,
    color="#247ba0",
)
plt.title("Charging cycles and battery health")
plt.xlabel("Total charging cycles (cycles)")
plt.ylabel("State of Health (%)")
plt.tight_layout()
plt.savefig("Reports/cycles_vs_soh.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(
    df[feature_columns + ["SoH_Percent"]].corr(),
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=-1,
    vmax=1,
)
plt.title("Correlation between model inputs and SoH")
plt.xlabel("Battery features and SoH")
plt.ylabel("Battery features and SoH")
plt.tight_layout()
plt.savefig("Reports/correlation_heatmap.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

status_counts = df["Battery_Status"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(status_counts.index, status_counts.values, color=["#2a9d8f", "#e76f51"])
plt.title("Battery status counts")
plt.xlabel("Status from Battery_Status")
plt.ylabel("Number of records (rows)")
plt.tight_layout()
plt.savefig("Reports/battery_status_counts.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(7, 6))
plt.pie(
    status_counts,
    labels=status_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#2a9d8f", "#e76f51"],
)
plt.title("Battery status share")
plt.tight_layout()
plt.savefig("Reports/battery_status_pie.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

battery_counts = df["Battery_Type"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(battery_counts.index, battery_counts.values, color="#247ba0")
plt.title("Records by battery type")
plt.xlabel("Battery type (category)")
plt.ylabel("Number of records (rows)")
plt.tight_layout()
plt.savefig("Reports/battery_type_bar.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("Status counts:", status_counts.to_dict())
print("The status labels are imbalanced; the rare label needs extra care.")

print("\nStage 5: Feature selection")
X = df[feature_columns]
y = df["SoH_Percent"]
X_class = df[feature_columns]
y_class = (df["Battery_Status"] == "Healthy").astype(int)
print("Regression inputs:", feature_columns)
print("Regression target: SoH_Percent (%)")
print("Classification inputs:", feature_columns)
print("Classification target: Battery_Status (1 = Healthy, 0 = Not Healthy)")
print("SoH_Percent is not a classification input. This prevents target leakage.")
print("Target leakage means giving the model information that already reveals its answer.")

split_summary = pd.DataFrame(
    {
        "Task": ["Regression", "Classification"],
        "Input columns": [len(feature_columns), len(feature_columns)],
        "Target": ["SoH_Percent (%)", "Battery_Status"],
    }
)
plt.figure(figsize=(9, 3))
plt.axis("off")
plt.table(
    cellText=split_summary.values,
    colLabels=split_summary.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Selected inputs and targets")
plt.tight_layout()
plt.savefig("Reports/feature_split.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 6: Train/test split")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class,
)
print("Regression training rows:", len(X_train))
print("Regression test rows:", len(X_test))
print("Classification training rows:", len(X_train_class))
print("Classification test rows:", len(X_test_class))
print("Not Healthy training rows:", int((y_train_class == 0).sum()))
print("Not Healthy test rows:", int((y_test_class == 0).sum()))

print("\nStage 7: Model training")
model1 = LinearRegression()
model1.fit(X_train[["Total_Charging_Cycles"]], y_train)
model2 = LinearRegression()
model2.fit(X_train, y_train)
model3 = Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "logistic",
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
    ]
)
model3.fit(X_train_class, y_train_class)
print(
    "Simple regression equation: SoH (%) = "
    f"{model1.coef_[0]:.6f} * cycles + {model1.intercept_:.6f}"
)
print("Multiple regression uses all five input columns.")
print("Logistic regression uses the same five inputs, scaled using training rows.")

simple_line = df.sort_values("Total_Charging_Cycles")
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Total_Charging_Cycles"],
    df["SoH_Percent"],
    s=12,
    alpha=0.3,
    color="#247ba0",
    label="Dataset rows",
)
plt.plot(
    simple_line["Total_Charging_Cycles"],
    model1.predict(simple_line[["Total_Charging_Cycles"]]),
    color="#e63946",
    linewidth=3,
    label="Fitted line",
)
plt.title("Simple Linear Regression: cycles and SoH")
plt.xlabel("Total charging cycles (cycles)")
plt.ylabel("State of Health (%)")
plt.legend()
plt.tight_layout()
plt.savefig("Reports/simple_regression_line.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 8: Model evaluation")
simple_train_prediction = model1.predict(
    X_train[["Total_Charging_Cycles"]]
)
simple_test_prediction = model1.predict(X_test[["Total_Charging_Cycles"]])
multiple_train_prediction = model2.predict(X_train)
multiple_test_prediction = model2.predict(X_test)
regression_results = pd.DataFrame(
    {
        "Model": ["Simple Linear", "Multiple Linear"],
        "MAE (%)": [
            mean_absolute_error(y_test, simple_test_prediction),
            mean_absolute_error(y_test, multiple_test_prediction),
        ],
        "MSE (% squared)": [
            mean_squared_error(y_test, simple_test_prediction),
            mean_squared_error(y_test, multiple_test_prediction),
        ],
        "RMSE (%)": [
            np.sqrt(mean_squared_error(y_test, simple_test_prediction)),
            np.sqrt(mean_squared_error(y_test, multiple_test_prediction)),
        ],
        "R2 (test)": [
            r2_score(y_test, simple_test_prediction),
            r2_score(y_test, multiple_test_prediction),
        ],
        "RSS (% squared)": [
            np.sum((y_test - simple_test_prediction) ** 2),
            np.sum((y_test - multiple_test_prediction) ** 2),
        ],
    }
)
print("Regression test results:")
print(regression_results.round(4).to_string(index=False))

train_test_results = pd.DataFrame(
    {
        "Model": ["Simple Linear", "Multiple Linear", "Logistic Regression"],
        "Train score": [
            model1.score(X_train[["Total_Charging_Cycles"]], y_train),
            model2.score(X_train, y_train),
            model3.score(X_train_class, y_train_class),
        ],
        "Test score": [
            model1.score(X_test[["Total_Charging_Cycles"]], y_test),
            model2.score(X_test, y_test),
            model3.score(X_test_class, y_test_class),
        ],
    }
)
print("\nTrain score versus test score:")
print(train_test_results.round(4).to_string(index=False))

classification_prediction = model3.predict(X_test_class)
classification_matrix = confusion_matrix(
    y_test_class,
    classification_prediction,
    labels=[0, 1],
)
classification_results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision (Not Healthy)",
            "Recall (Not Healthy)",
            "F1 (Not Healthy)",
        ],
        "Test value": [
            accuracy_score(y_test_class, classification_prediction),
            precision_score(
                y_test_class,
                classification_prediction,
                pos_label=0,
                zero_division=0,
            ),
            recall_score(
                y_test_class,
                classification_prediction,
                pos_label=0,
                zero_division=0,
            ),
            f1_score(
                y_test_class,
                classification_prediction,
                pos_label=0,
                zero_division=0,
            ),
        ],
    }
)
print("\nClassification test results:")
print(classification_results.round(4).to_string(index=False))
print("Confusion matrix; rows = actual, columns = predicted; order = Not Healthy, Healthy:")
print(classification_matrix)
print("Classification scores are uncertain because only four rows are Not Healthy.")

plt.figure(figsize=(11, 4))
plt.axis("off")
plt.table(
    cellText=regression_results.round(4).values,
    colLabels=regression_results.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Regression test metrics")
plt.tight_layout()
plt.savefig("Reports/model_results.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 3))
plt.axis("off")
plt.table(
    cellText=train_test_results.round(4).values,
    colLabels=train_test_results.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Train score and test score")
plt.tight_layout()
plt.savefig("Reports/train_test_scores.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 3))
plt.axis("off")
plt.table(
    cellText=classification_results.round(4).values,
    colLabels=classification_results.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Classification test metrics")
plt.tight_layout()
plt.savefig("Reports/classification_metrics.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(
    classification_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Healthy", "Healthy"],
    yticklabels=["Not Healthy", "Healthy"],
)
plt.title("Confusion matrix")
plt.xlabel("Predicted status")
plt.ylabel("Actual status")
plt.tight_layout()
plt.savefig("Reports/confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

folds = KFold(n_splits=5, shuffle=True, random_state=42)
simple_cv = cross_val_score(
    LinearRegression(),
    X[["Total_Charging_Cycles"]],
    y,
    cv=folds,
    scoring="r2",
)
multiple_cv = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=folds,
    scoring="r2",
)
classification_cv = cross_val_score(
    Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            ),
        ]
    ),
    X_class,
    y_class,
    cv=folds,
    scoring="accuracy",
)
cv_results = pd.DataFrame(
    {
        "Model and score": [
            "Simple Linear R2",
            "Multiple Linear R2",
            "Logistic accuracy",
        ],
        "Fold 1": [simple_cv[0], multiple_cv[0], classification_cv[0]],
        "Fold 2": [simple_cv[1], multiple_cv[1], classification_cv[1]],
        "Fold 3": [simple_cv[2], multiple_cv[2], classification_cv[2]],
        "Fold 4": [simple_cv[3], multiple_cv[3], classification_cv[3]],
        "Fold 5": [simple_cv[4], multiple_cv[4], classification_cv[4]],
        "Mean": [
            simple_cv.mean(),
            multiple_cv.mean(),
            classification_cv.mean(),
        ],
    }
)
print("\nFive-fold cross-validation scores:")
print(cv_results.round(4).to_string(index=False))

plt.figure(figsize=(11, 3))
plt.axis("off")
plt.table(
    cellText=cv_results.round(4).values,
    colLabels=cv_results.columns,
    cellLoc="center",
    loc="center",
)
plt.title("Five-fold cross-validation")
plt.tight_layout()
plt.savefig("Reports/cross_validation.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 9: Prediction")
sample = X_test.iloc[[0]]
sample_actual = y_test.iloc[0]
sample_prediction = model2.predict(sample)[0]
print("Held-out sample actual SoH (%):", f"{sample_actual:.2f}")
print("Held-out sample predicted SoH (%):", f"{sample_prediction:.2f}")

plt.figure(figsize=(7, 5))
plt.bar(
    ["Actual SoH (%)", "Predicted SoH (%)"],
    [sample_actual, sample_prediction],
    color=["#2a9d8f", "#e9c46a"],
)
plt.title("One held-out battery: actual and predicted SoH")
plt.xlabel("SoH value")
plt.ylabel("State of Health (%)")
plt.ylim(0, 105)
plt.text(0, sample_actual + 1.5, f"{sample_actual:.2f} %", ha="center", fontweight="bold")
plt.text(1, sample_prediction + 1.5, f"{sample_prediction:.2f} %", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("Reports/sample_prediction.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 10: Deployment with user input")
print("Enter five battery measurements from the dataset's named columns.")
age_input = float(input("Vehicle age (months): "))
cycles_input = float(input("Total charging cycles (cycles): "))
temperature_input = float(input("Average temperature (°C): "))
charge_input = float(input("Fast-charge ratio (0 to 1): "))
discharge_input = float(input("Average discharge rate (C-rate): "))
new_battery = pd.DataFrame(
    [
        [
            age_input,
            cycles_input,
            temperature_input,
            charge_input,
            discharge_input,
        ]
    ],
    columns=feature_columns,
)
status_prediction = int(model3.predict(new_battery)[0])
healthy_probability = model3.predict_proba(new_battery)[0][1] * 100
soh_prediction = float(np.clip(model2.predict(new_battery)[0], 0, 100))

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

print("Battery status:", battery_status)
print("Predicted State of Health:", f"{soh_prediction:.2f} %")
print("Probability of Healthy:", f"{healthy_probability:.2f} %")
print("Advice:", advice)
print("Learning project on synthetic data; not for real safety decisions.")

plt.figure(figsize=(7, 5))
plt.bar(
    ["Predicted SoH (%)"],
    [soh_prediction],
    color="#247ba0",
)
plt.title("User-input battery prediction")
plt.xlabel("Prediction result")
plt.ylabel("State of Health (%)")
plt.ylim(0, 105)
plt.text(0, soh_prediction + 1.5, f"{soh_prediction:.2f} %", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("Reports/deployment_prediction.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nStage 11: Monitoring")
print("Compare future predictions with measured outcomes and check errors.")
print("Ask the teacher before using a non-synthetic dataset.")
