
# 📘 **Stock Price Prediction Using Linear & Polynomial Regression**

This project is a **beginner-friendly Machine Learning model** designed to predict stock prices using **Linear Regression** and **Polynomial Regression**.
It follows concepts typically taught in college courses such as:

* Linear Regression
* Polynomial Regression
* Python basics
* Numpy, Pandas, Matplotlib
* Scikit-Learn models & evaluation metrics

The objective is to build a simple predictive model for stock prices and compare the performance of linear and polynomial models.

---

# 📂 **Project Overview**

This project uses four stocks:

* **AMZN** (Amazon)
* **DPZ** (Domino’s Pizza)
* **BTC** (Bitcoin)
* **NFLX** (Netflix)

A lag-1 feature (yesterday’s price) is used to predict the **next day’s closing price**.

Two regression models are trained:

1. **Linear Regression**
2. **Polynomial Regression (Degree 3)**

Both models are evaluated and compared using metrics such as:

* **RMSE** – Root Mean Squared Error
* **R² Score** – Goodness-of-fit
* **Accuracy Score** – Direction accuracy (up/down movement)
* **MAE / MSE**

The results are plotted and displayed for all four stocks.

---

# 🧠 **Features of This Project**

### ✔ Linear & Polynomial Regression

Builds and compares two fundamental ML models.

### ✔ Lag Feature Engineering

Uses **lag-1**: yesterday’s price → predict tomorrow’s price.

### ✔ Model Evaluation

Includes:

* RMSE
* MAE
* R² Score
* Up/Down Accuracy (custom accuracy for regression)

### ✔ Visual Comparison

Plots Actual vs Predicted prices for each stock.

### ✔ User Input Prediction

Allows the user to input a price and get next-day predictions:

```
Enter stock name: AMZN
Enter yesterday's price: 1830
```

---

# 📊 **Technologies Used**

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Python       | Main programming language   |
| Pandas       | Data loading & cleaning     |
| NumPy        | Numerical operations        |
| Matplotlib   | Data visualization          |
| Scikit-Learn | Regression models & metrics |

---

# 📝 **How the Model Works**

### **1. Data Loading**

The dataset (`portfolio_data.csv`) contains:

* Date
* AMZN
* DPZ
* BTC
* NFLX

### **2. Preprocessing**

* Dates sorted in ascending order
* Missing values handled
* Lag-1 feature created

### **3. Train/Test Split**

80% → Training
20% → Testing

### **4. Model Training**

Two models are trained separately for each stock:

* Linear Regression
* Polynomial Regression (Degree 3)

### **5. Model Evaluation**

We compute:

* RMSE
* MAE
* R² Score
* Direction Accuracy

### **6. Visualization**

Actual vs Predicted prices are plotted for all four stocks.

### **7. User Input Prediction**

The user can predict future prices using both models.

---

# 📈 **Sample Output Metrics**

| Stock | Linear RMSE | Polynomial RMSE | Better Model |
| ----- | ----------- | --------------- | ------------ |
| AMZN  | ✔ Lower     | Higher          | Linear       |
| DPZ   | ✔ Lower     | Higher          | Linear       |
| BTC   | ✔ Lower     | Higher          | Linear       |
| NFLX  | ✔ Lower     | Higher          | Linear       |

**Conclusion:**
Linear Regression performed better for all four stocks.

---

# ▶️ **How to Run the Project**

### **1. Install dependencies**

```bash
pip install pandas numpy matplotlib scikit-learn
```

### **2. Place the dataset**

Ensure `portfolio_data.csv` is in the same folder.

### **3. Run the Python script**

```bash
python stock_prediction.py
```

### **4. Predict future price**

After training, the program lets you enter:

* Stock name
* Yesterday’s price

It returns:

* Linear Prediction
* Polynomial Prediction

---

# 📁 **Project Structure**

```
📦 stock-price-prediction/
├── 📄 portfolio_data.csv
├── 📄 stock_prediction.py
├── 📄 README.md   ← (This file)
└── 📁 plots/      ← Generated graphs
```

---

# 🧩 **Future Improvements**

✔ Add more lag features (lag-3, lag-7, lag-30)
✔ Add technical indicators (MA, RSI, EMA)
✔ Train advanced models (LSTM, Random Forest, XGBoost)
✔ Convert into a Streamlit web app

---

# 🙌 **Created By**

A student learning:

* Machine Learning
* Python
* Regression
* Data Preprocessing
