# 🏠 House Price Prediction using Linear Regression

A Machine Learning project that predicts house prices using **Linear Regression** based on property features such as area, bedrooms, bathrooms, stories, parking, and other categorical attributes.

This project demonstrates the complete machine learning workflow — from **data preprocessing and exploratory data analysis (EDA)** to **model training, evaluation, visualization, and prediction**.

---

## 📌 Project Overview

House prices depend on several factors such as property area, number of bedrooms, bathrooms, location-related features, parking, furnishing status, and other amenities.

In this project, **Linear Regression** is used to learn the relationship between these features and house prices.

The project covers:

* Data loading and inspection
* Data cleaning
* Exploratory Data Analysis (EDA)
* Handling categorical variables
* One-Hot Encoding
* Train-Test Split
* Linear Regression model training
* House price prediction
* Model evaluation
* Actual vs Predicted visualization
* Residual analysis
* Prediction for a new house
* Saving the trained model

---

## 🎯 Objective

The main objective of this project is to:

> **Build a Linear Regression model that can predict house prices based on property characteristics.**

---

## 📊 Dataset

The project uses a **Housing Dataset** containing information about residential properties.

### Dataset Features

| Feature            | Description                                     |
| ------------------ | ----------------------------------------------- |
| `area`             | Area of the house                               |
| `bedrooms`         | Number of bedrooms                              |
| `bathrooms`        | Number of bathrooms                             |
| `stories`          | Number of stories                               |
| `mainroad`         | Whether the house is connected to the main road |
| `guestroom`        | Whether the house has a guest room              |
| `basement`         | Whether the house has a basement                |
| `hotwaterheating`  | Whether hot water heating is available          |
| `airconditioning`  | Whether air conditioning is available           |
| `parking`          | Number of parking spaces                        |
| `prefarea`         | Whether the house is in a preferred area        |
| `furnishingstatus` | Furnishing status of the house                  |
| `price`            | Target variable — house price                   |

### Dataset Size

* **Rows:** 545
* **Original Features:** 13
* **Target Variable:** `price`

---

## 🛠️ Technologies & Libraries

The project is implemented using **Python**.

### Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Predictions
   ↓
Model Evaluation
   ↓
Visualization
   ↓
New House Prediction
   ↓
Save Model
```

---

## 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Scikit-learn is used for model training, data splitting, and evaluation.

---

## 2️⃣ Load the Dataset

```python
df = pd.read_csv("Housing.csv")
```

Basic dataset inspection:

```python
df.head()
df.shape
df.info()
df.describe()
```

---

## 3️⃣ Data Cleaning

Missing values were checked using:

```python
df.isnull().sum()
```

Duplicate records were checked using:

```python
df.duplicated().sum()
```

The dataset was then prepared for machine learning.

---

## 4️⃣ Exploratory Data Analysis

Several visualizations were created to understand the dataset.

### Price Distribution

```python
sns.histplot(df["price"], kde=True)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("House Price Distribution")
plt.show()
```

### Area vs Price

```python
sns.scatterplot(x=df["area"], y=df["price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs House Price")
plt.show()
```

### Bedrooms vs Price

```python
sns.boxplot(x=df["bedrooms"], y=df["price"])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs House Price")
plt.show()
```

### Correlation Analysis

A correlation heatmap was used to understand relationships between numerical variables.

---

## 5️⃣ Feature and Target Separation

The target variable is `price`.

```python
X = df.drop("price", axis=1)
y = df["price"]
```

Where:

* `X` → Input features
* `y` → Target variable

---

## 6️⃣ Categorical Encoding

Machine learning models require numerical input.

Categorical columns such as:

```text
mainroad
guestroom
basement
hotwaterheating
airconditioning
prefarea
furnishingstatus
```

were converted into numerical values using **One-Hot Encoding**.

```python
categorical_columns = X.select_dtypes(include=["object"]).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

X = X.astype(int)
```

After encoding, the dataset contained **13 numerical features**.

---

## 7️⃣ Train-Test Split

The dataset was divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

### Dataset Split

* **80% → Training data**
* **20% → Testing data**

The model learns from the training data and is evaluated on unseen testing data.

---

## 8️⃣ Build the Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

The model learns the relationship between the house features and their prices.

---

## 9️⃣ Make Predictions

Predictions were generated using the test dataset:

```python
y_pred = model.predict(X_test)
```

A comparison between actual and predicted prices:

```python
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(results.head(10))
```

---

## 📈 Model Evaluation

The model was evaluated using four commonly used regression metrics:

* MAE
* MSE
* RMSE
* R² Score

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
```

### 📊 Results

| Metric   |               Result |
| -------- | -------------------: |
| MAE      |          ₹970,043.40 |
| MSE      | 1,754,318,687,330.66 |
| RMSE     |        ₹1,324,506.96 |
| R² Score |               0.6529 |

### Interpretation

**MAE ≈ ₹9.70 lakh**

On average, the model's prediction differs from the actual house price by approximately ₹9.70 lakh.

**RMSE ≈ ₹13.25 lakh**

RMSE gives more importance to larger prediction errors.

**R² ≈ 0.6529**

The model explains approximately **65.29% of the variation** in house prices on the test dataset.

> Note: R² is not the same as prediction accuracy.

---

## 📉 Actual vs Predicted Prices

The actual and predicted prices can be visualized using a scatter plot.

```python
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()
```

### Interpretation

Each point represents a house.

* X-axis → Actual price
* Y-axis → Predicted price
* Points closer to the diagonal line indicate predictions closer to the actual prices.

---

## 📊 Residual Analysis

Residuals represent the difference between actual and predicted values.

```python
residuals = y_test - y_pred
```

Residual plot:

```python
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()
```

Residual analysis helps identify patterns in the model's prediction errors.

---

## 🏡 Predict Price of a New House

The trained model can also be used to predict the price of a new property.

Example:

```python
new_house = pd.DataFrame({
    "area": [6000],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "mainroad": ["yes"],
    "guestroom": ["no"],
    "basement": ["yes"],
    "hotwaterheating": ["no"],
    "airconditioning": ["yes"],
    "parking": [2],
    "prefarea": ["yes"],
    "furnishingstatus": ["semi-furnished"]
})
```

The same preprocessing used during training must be applied to the new data.

```python
new_house = pd.get_dummies(
    new_house,
    columns=categorical_columns,
    drop_first=True
)

new_house = new_house.reindex(
    columns=X.columns,
    fill_value=0
)
```

Prediction:

```python
predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])
```

---

## 💾 Save the Trained Model

The trained model can be saved using Joblib.

```python
import joblib

joblib.dump(
    model,
    "house_price_linear_regression.pkl"
)
```

The saved model can later be loaded without retraining:

```python
loaded_model = joblib.load(
    "house_price_linear_regression.pkl"
)
```

---

## 📁 Project Structure

```text
house-price-prediction/
│
├── Housing.csv
├── house_price_prediction.ipynb
├── house_price_linear_regression.pkl
├── README.md
└── requirements.txt
```

---

## 📦 Requirements

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project

```bash
cd house-price-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
house_price_prediction.ipynb
```

Run the notebook cells sequentially.

---

## 🔮 Future Improvements

The current Linear Regression model provides a baseline solution. The project can be improved by:

* Feature engineering
* Feature scaling where appropriate
* Log transformation of the target variable
* Cross-validation
* Hyperparameter tuning
* Regularized regression such as Ridge and Lasso
* Trying tree-based models
* Comparing multiple regression algorithms
* Building a prediction web application
* Using a preprocessing `Pipeline` and `ColumnTransformer`
* Deploying the model using Flask, FastAPI, or Streamlit

---

## 🧠 Key Concepts Learned

Through this project, the following Machine Learning concepts were implemented:

* Supervised Learning
* Regression
* Linear Regression
* Feature and Target Separation
* Data Cleaning
* Exploratory Data Analysis
* One-Hot Encoding
* Train-Test Split
* Model Training
* Model Prediction
* MAE
* MSE
* RMSE
* R² Score
* Residual Analysis
* Model Serialization

---

## 📌 Conclusion

This project demonstrates an end-to-end **Machine Learning regression workflow** for predicting house prices.

Linear Regression was trained using property-related features and evaluated on unseen test data. The model achieved an **R² score of 0.6529** on the test set, providing a baseline for further experimentation and model improvement.

The project can be extended by comparing different regression algorithms and developing a deployable house price prediction application.

---

## 👨‍💻 Author

**Atharva Pundkar**

* GitHub: [atharva270405](https://github.com/atharva270405)
* LinkedIn: [Atharva Pundkar](https://www.linkedin.com/in/atharva-pundkar-077810235/)
