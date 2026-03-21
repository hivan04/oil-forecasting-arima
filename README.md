# Forecasting Crude Oil Futures Spread (ARIMA vs ARIMAX Models)

This project investigates whether econometric models can forecast the **WTI crude oil futures spread (CL2–CL1)** and whether such forecasts can generate **economically meaningful trading strategies**.

---

## 📌 Overview

Crude oil futures spreads reflect underlying market conditions such as supply-demand dynamics, storage costs, and expectations of future prices. Unlike spot prices, these spreads may exhibit **predictable structure**, making them suitable for time-series modelling.

This project compares two models:

- **ARIMA (baseline model)**
- **ARIMAX (with macroeconomic & financial variables)**

We evaluate both **statistical forecasting performance** and **real-world trading profitability**.

---

## 🎯 Objectives

- Forecast the oil futures spread using time-series models  
- Evaluate whether **exogenous variables improve predictions**  
- Test whether forecasts can be translated into a **profitable trading strategy**

---


---

## 📊 Data

- Monthly data frequency  
- Variables include:
  - WTI futures prices (CL1, CL2)
  - Macroeconomic indicators (CPI, yields)
  - Market variables (S&P 500, VIX, DXY)

### Target Variable

\[
\text{Log Spread} = \ln(CL2) - \ln(CL1)
\]

- Captures **contango vs backwardation**
- Log transformation improves **stationarity and stability** :contentReference[oaicite:1]{index=1}

---

## ⚙️ Methodology

### 1. Data Preparation
- Monthly alignment of all variables  
- Log transformation of spread  
- 70:30 **train-test split (time-series consistent)**  

### 2. Models

#### 🔹 ARIMA (Baseline)
- Selected: **ARIMA(1,1,1)**
- Captures autoregressive dynamics of the spread

#### 🔹 ARIMAX
- Includes:
  - CPI
  - Interest rates
  - S&P 500
  - USD index (DXY)
  - VIX
- Selected: **ARIMAX(2,1,0)**

⚠️ Note: Sample size reduced significantly due to data alignment (~85% reduction), impacting performance :contentReference[oaicite:2]{index=2}

---

### 3. Model Evaluation

#### Statistical Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Hit Rate
- Out-of-sample \(R^2\)

#### Forecasting Approaches
- Non-rolling forecasts  
- Rolling window forecasts (24-month window)

---

## 📈 Trading Strategy

A simple signal-based strategy:

- **Long** if forecasted spread > current spread  
- **Short** otherwise  

\[
Signal_{t+1} =
\begin{cases}
+1 & \hat{S}_{t+1} > S_t \\
-1 & \text{otherwise}
\end{cases}
\]

Performance evaluated via:
- Cumulative returns  
- Sharpe ratio  
- Drawdowns  

---

## 🧠 Key Findings

### 📊 Forecasting Performance
- ARIMA consistently outperforms ARIMAX out-of-sample  
- ARIMAX improves in-sample fit but **fails to generalise**

### 📉 Rolling Forecast Results
- ARIMA delivers:
  - Lower RMSE  
  - Higher directional accuracy (~63%) :contentReference[oaicite:3]{index=3}  
  - More stable predictions  

### 💰 Trading Performance
- ARIMA strategy:
  - Higher cumulative returns  
  - Sharpe ratio > 1  
  - Lower drawdowns  

- ARIMAX strategy:
  - Poor profitability  
  - High turnover  
  - Large drawdowns  

---

## ⚠️ Key Insight

> Simpler models outperform more complex ones when data is limited.

- ARIMAX suffers from **small sample size (54 observations)**  
- Added variables introduce **noise instead of signal**  
- ARIMA provides **robust and tradeable forecasts** 

