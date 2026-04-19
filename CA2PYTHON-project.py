import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------- Data Loading & Understanding ----------------
# Load dataset and explore basic structure

data = pd.read_csv("/Users/sarthak/Downloads/populationgroup-wise-deposits.csv")

print(data.head())
print(data.info())
print(data.describe())

# ---------------- Data Cleaning & Preprocessing ----------------
# Remove missing values and duplicates, fix data types

data = data.dropna()
data = data.drop_duplicates()

data['year'] = data['year'].astype(int)
data['no_of_offices'] = data['no_of_offices'].astype(int)
data['no_of_accounts'] = data['no_of_accounts'].astype(int)
data['deposit_amount'] = data['deposit_amount'].astype(float)

# ---------------- Objective 1 ----------------
# Analyze deposit distribution across population groups

plt.figure(figsize=(10,6))

sns.boxplot(x="population_group", y="deposit_amount", data=data)

plt.title("Deposit Distribution across Population Groups")
plt.xlabel("Population Group")
plt.ylabel("Deposit Amount")

plt.show()





# -------------------- Objective 2--------------
#Region-wise Deposit Contribution


region_data = data.groupby("region")["deposit_amount"].sum()

plt.figure(figsize=(8,8))

plt.pie(region_data, labels=region_data.index, autopct='%1.1f%%')

plt.title("Region-wise Deposit Contribution")

plt.show()


#------------------------- Objective 3------------------
# Top 10 States by Deposit


state_data = data.groupby("state_name")["deposit_amount"].sum() \
                 .sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))

sns.barplot(x=state_data.values, y=state_data.index,
            hue=state_data.index, palette="magma", legend=False)

plt.title("Top 10 States by Deposit")
plt.xlabel("Deposit Amount")
plt.ylabel("State")

plt.show()


# ------------- Objective 4-------------------------------
# Offices vs Accounts Relationship


plt.figure(figsize=(12,7))

sns.scatterplot(x=data["no_of_offices"],
                y=data["no_of_accounts"],
                hue=data["region"],
                palette="Set1")

plt.title("Offices vs Accounts (Region-wise)")
plt.xlabel("Number of Offices")
plt.ylabel("Number of Accounts")

plt.tight_layout()
plt.show()



# ----------------------- Objective 6---------------------
#Correlation Analysis
numeric_cols = ['no_of_offices', 'no_of_accounts', 'deposit_amount']

plt.figure(figsize=(8,6))

sns.heatmap(data[numeric_cols].corr(),
            annot=True,
            cmap='coolwarm',
            linewidths=1,
            linecolor='black')

plt.title("Correlation Matrix")
plt.show()
# ----------------- Objective 5---------------------------
# Machine Learning (Linear Regression)



scaler = MinMaxScaler()

data[['no_of_accounts','no_of_offices','deposit_amount']] = scaler.fit_transform(
    data[['no_of_accounts','no_of_offices','deposit_amount']]
)
X = data[['no_of_accounts']]
y = data[['deposit_amount']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34
)
model = LinearRegression()
model.fit(X_train, y_train)
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=3)
plt.xlabel('Number of Accounts')
plt.ylabel('Deposit Amount')
plt.title('Linear Regression Fit')
plt.show()
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")


# Actual vs Predicted Visualization


plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot([0,1], [0,1], color='red', linestyle='--')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.show()