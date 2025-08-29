# Starbucks Sales Analytics & Forecasting (Synthetic Data)

## 📌 Overview
This project is a **Python-based analytics pipeline** that loads, cleans, and analyzes **synthetic Starbucks sales data**.  
It was built as a portfolio project to demonstrate **data wrangling, exploratory data analysis (EDA), and forecasting skills** using modern Python libraries.  

The program processes raw CSVs (`menu.csv`, `customers.csv`, `sales.csv`), validates and cleans them, generates insights, and saves cleaned datasets + visualizations into an organized output folder.

---

## 🚀 Features
- **Data Cleaning & Validation**
  - Detects missing values, duplicates, and incorrect data types.
  - Cleans and converts columns (dates, numerics, booleans).
  - Saves cleaned versions of all datasets.

- **Exploratory Data Analysis (EDA)**
  - Top selling items by quantity & revenue.
  - Daily, hourly, and weekday sales patterns.
  - Revenue by **channel** (in-store vs mobile) and **payment method**.
  - Loyalty membership impact on revenue and average order value (AOV).

- **Forecasting**
  - Linear trend-based daily revenue forecast.
  - Weekday effect forecast (captures cyclical patterns).

- **Customer Segmentation**
  - Basic **RFM analysis** (Recency, Frequency, Monetary value).
  - Segments customers for marketing insights.

- **Automated Outputs**
  - All cleaned datasets saved as `cleaned_*.csv`.
  - All plots saved directly into `starbucks_sales_analytics_outputs/`.

---

## 🛠️ Tech Stack
- **Language:** Python 3  
- **Libraries:**  
  - `pandas` – data wrangling  
  - `numpy` – numerical operations  
  - `matplotlib` – data visualization  
  - `scikit-learn` – forecasting (linear regression)  

---

## 📊 Example Visuals
The program generates professional charts such as:

- **Top 10 Items by Revenue**
- **Daily Revenue Trends**
- **Revenue by Channel & Payment Method**
- **Average Order Value by Loyalty Status**
- **Forecasted Revenue (Linear + Weekday Effects)**
- **Customer Segmentation (RFM)**

*(see `/starbucks_sales_analytics_outputs/` for all charts)*

---

## ▶️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/starbucks-sales-analytics.git
   cd starbucks-sales-analytics
2. Install dependencies: pip install -r requirements.txt
3. Run the program: python starbucks_sales_analytics.py
4. Outputs will be saved in: /starbucks_sales_analytics_outputs/
