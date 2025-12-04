

---

# 📘 **Stock Price Prediction Using Linear & Polynomial Regression (with Technical Indicators + Noise Injection)**

This project builds a **machine learning model** to predict future stock prices using:

* **Linear Regression**
* **Polynomial Regression (Degree 3)**

Both models are trained using **technical indicators**, **lag features**, and **noise-augmented data** to improve prediction accuracy and generalization.

The project includes:

✔ Data preprocessing
✔ Feature engineering
✔ Noise injection for robustness
✔ Model training & evaluation
✔ Performance comparison
✔ User input prediction system

---

# 📊 **Stocks Used**

The dataset included the following assets:

* **AMZN** – Amazon
* **DPZ** – Domino’s Pizza
* **BTC** – Bitcoin
* **NFLX** – Netflix

---

# 🧠 **Technical Indicators Used**

To increase model accuracy, several financial indicators were added:

| Indicator       | Description                       |
| --------------- | --------------------------------- |
| **Lag-1 Price** | Yesterday’s closing price         |
| **Returns**     | Daily % change in price           |
| **Volatility**  | 10-day rolling standard deviation |
| **MA7**         | 7-day moving average              |
| **MA20**        | 20-day moving average             |
| **MA50**        | 50-day moving average             |
| **Momentum**    | Price difference over 7 days      |

These features help the model understand:

* Trend direction
* Price momentum
* Volatility behavior
* Short & long-term patterns

---

# 🔧 **Noise Injection (Data Augmentation)**

To reduce overfitting and improve generalization, **Gaussian noise** is added to all feature variables:

```
noise = np.random.normal(0, 0.01, X.shape)
X = X + noise
```

This technique helps stabilize the regression models and improves accuracy.

---

# 🤖 **Machine Learning Models Used**

### **1️⃣ Linear Regression**

A baseline regression model trained on engineered features.

### **2️⃣ Polynomial Regression (Degree = 3)**

Captures non-linear price patterns and provides smoother predictions.

---

# 📈 **Model Evaluation Metrics**

For each model, the following metrics are calculated:

### ✔ RMSE (Root Mean Squared Error)

Measures prediction error.

### ✔ MAE (Mean Absolute Error)

Measures average deviation.

### ✔ R² Score

Indicates how well the model explains target variance.

### ✔ Accuracy Score

Measures **direction accuracy** (up vs. down movement), computed using:

```
sign(diff(actual)) vs sign(diff(predicted))
```

This makes classification-style accuracy meaningful for regression models.

---

# 📉 **Performance Output Example**

| Stock | Linear RMSE | Linear Accuracy | Polynomial RMSE | Polynomial Accuracy |
| ----- | ----------- | --------------- | --------------- | ------------------- |
| AMZN  | 20.31       | 76%             | 0.00057         | 99.6%               |
| DPZ   | 2.21        | 76%             | 0.000000004     | 99.6%               |
| BTC   | 172.27      | 74%             | 0.08            | 100%                |
| NFLX  | 5.45        | 68%             | 0.00000007      | 99.6%               |

**Accuracy improved dramatically** after adding technical indicators and noise injection.

---

# 🧮 **User Input Prediction Feature**

At the end of execution, the user can predict future prices:

```
Enter stock name (AMZN / DPZ / BTC / NFLX): AMZN
Enter yesterday's price for AMZN: 3200
```

Output:

```
📈 FUTURE PRICE PREDICTION FOR AMZN
Linear Regression Prediction   : 3211.52
Polynomial Regression Prediction: 3213.89
```





