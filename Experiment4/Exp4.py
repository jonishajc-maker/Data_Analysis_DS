import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ---------------- STUDENT PERFORMANCE ----------------

student = pd.read_csv("Student_Performance_Dataset.csv")

print("Student Dataset")
print(student.info())
print("Missing values:\n", student.isnull().sum())
print("Duplicates:", student.duplicated().sum())

# Fill missing values
student["Gender"] = student["Gender"].fillna(student["Gender"].mode()[0])
for col in ["Attendance", "Study_Hours", "Assignment_Score"]:
    student[col] = student[col].fillna(student[col].median())

# Remove duplicates
student = student.drop_duplicates()

# Encode Gender
student["Gender"] = LabelEncoder().fit_transform(student["Gender"])

# Remove outliers using IQR
for col in ["Study_Hours", "Final_Marks"]:
    Q1 = student[col].quantile(0.25)
    Q3 = student[col].quantile(0.75)
    IQR = Q3 - Q1
    student = student[
        (student[col] >= Q1 - 1.5 * IQR) &
        (student[col] <= Q3 + 1.5 * IQR)
    ]

# Min-Max normalization
num_cols = ["Age", "Attendance", "Study_Hours",
            "Assignment_Score", "Internal_Marks", "Final_Marks"]

scaler = MinMaxScaler()
student[num_cols] = scaler.fit_transform(student[num_cols])

# Standardization
standard = StandardScaler()
student[num_cols] = standard.fit_transform(student[num_cols])

student.to_csv("student_preprocessed.csv", index=False)


# ---------------- HOUSE PRICE ----------------

house = pd.read_csv("Housing.csv")

print("\nHouse Price Dataset")
print(house.isnull().sum())
print("Duplicates:", house.duplicated().sum())

house = house.drop_duplicates()

# Encode categorical columns
house["airconditioning"] = LabelEncoder().fit_transform(
    house["airconditioning"]
)
house["furnishingstatus"] = LabelEncoder().fit_transform(
    house["furnishingstatus"]
)

# Remove outliers from area and price
for col in ["area", "price"]:
    Q1 = house[col].quantile(0.25)
    Q3 = house[col].quantile(0.75)
    IQR = Q3 - Q1
    house = house[
        (house[col] >= Q1 - 1.5 * IQR) &
        (house[col] <= Q3 + 1.5 * IQR)
    ]

# Min-Max normalization
cols = ["area", "bedrooms", "bathrooms", "price"]
house[cols] = MinMaxScaler().fit_transform(house[cols])

house.to_csv("houseprice_cleaned.csv", index=False)


# ---------------- HEALTHCARE ----------------

health = pd.read_csv("Healthcare_Patient_Dataset.csv")

print("\nHealthcare Dataset")
print(health.isnull().sum())

# Fill missing values
num_cols = health.select_dtypes(include="number").columns
cat_cols = health.select_dtypes(include="object").columns

for col in num_cols:
    health[col] = health[col].fillna(health[col].median())

for col in cat_cols:
    health[col] = health[col].fillna(health[col].mode()[0])

# Encode Gender and Diagnosis
encoder = LabelEncoder()

health["Gender"] = encoder.fit_transform(health["Gender"])
health["Diagnosis"] = encoder.fit_transform(health["Diagnosis"])

# Normalize numerical values
num_cols = ["Age", "Height", "Weight", "Blood_Pressure", "Blood_Sugar"]
health[num_cols] = MinMaxScaler().fit_transform(health[num_cols])

# Correlation matrix
print("\nCorrelation Matrix:")
print(health[num_cols].corr())

health.to_csv("healthcare_processed.csv", index=False)


# ---------------- ONLINE RETAIL ----------------

retail = pd.read_csv("Online Retail.csv")

print("\nOnline Retail Dataset")
print("Duplicates:", retail.duplicated().sum())

retail = retail.drop_duplicates()

# Missing CustomerID
retail["CustomerID"] = retail["CustomerID"].fillna(
    retail["CustomerID"].median()
)

retail["InvoiceDate"] = pd.to_datetime(retail["InvoiceDate"])

# One-hot encoding Country
retail = pd.get_dummies(retail, columns=["Country"], dtype=int)

# Remove outliers
for col in ["Quantity", "UnitPrice"]:
    Q1 = retail[col].quantile(0.25)
    Q3 = retail[col].quantile(0.75)
    IQR = Q3 - Q1
    retail = retail[
        (retail[col] >= Q1 - 1.5 * IQR) &
        (retail[col] <= Q3 + 1.5 * IQR)
    ]

# Normalize
retail[["Quantity", "UnitPrice"]] = MinMaxScaler().fit_transform(
    retail[["Quantity", "UnitPrice"]]
)

retail.to_csv("retail_sales_cleaned.csv", index=False)

print("\nExperiment 4 completed successfully!")