"""
Starbucks Sales Analytics & Forecasting (Synthetic Data)
Author: Anthony "AJ" Clark

This program loads, cleans, analyzes, and visualizes synthetic Starbucks-style sales data. 
It is designed as a portfolio project to demonstrate Python, Pandas, and data science skills.

Features:
- Loads CSV datasets (menu, customers, sales) into Pandas DataFrames.
- Performs data validation: shapes, dtypes, missing values, duplicates.
- Cleans and normalizes data (numeric conversions, date/time parsing, categorical fixes).
- Explores business questions with descriptive stats and visualizations:
  - Top-selling items
  - Daily and hourly revenue trends
  - Revenue by channel, payment method, and loyalty status
- Builds customer-level RFM analysis for segmentation.
- Implements forecasting:
  - Rolling averages
  - Linear regression
  - Regression with weekday seasonality
- Generates summary tables and charts for business insights.

Requirements:
- Python libraries: pandas, numpy, matplotlib, scikit-learn
"""

# Section 1: Imports and Data Loading
import pandas as pd

# Load data from CSV files
menu = pd.read_csv("menu.csv")
customers = pd.read_csv("customers.csv")
sales = pd.read_csv("sales.csv")

# Data Confirmation: Print the number of rows and columns of each DataFrame
print("Menu shape:", menu.shape)
print("Customers shape:", customers.shape)
print("Sales shape:", sales.shape)

# Section 2: Data and Info Checks
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# Print schema info for each dataset
print("\n--- MENU.INFO ---")
menu.info()
print("\n--- CUSTOMERS.INFO ---")
customers.info()
print("\n--- SALES.INFO ---")
sales.info()

# Preview the first few rows from each dataset
print("\nMENU HEAD:")
print(menu.head())
print("\nCUSTOMERS HEAD:")
print(customers.head())
print("\nSALES HEAD:")
print(sales.head())

# Columns expected to be numeric or boolean
expected_numeric_menu = ["item_price"]
expected_numeric_sales = ["quantity", "unit_price", "discount", "total_price"]
expected_bool_customers = ["is_member"]

def check_numeric(df, cols, df_name):
    for c in cols:
        if c in df.columns:
            status = "OK (numeric)" if is_numeric_dtype(df[c]) else f"NOT numeric (dtype={df[c].dtype})"
        else:
            status = "MISSING COLUMN"
        print(f"{df_name}.{c}: {status}")

def check_bool(df, cols, df_name):
    for c in cols:
        if c in df.columns:
            status = "OK (bool)" if is_bool_dtype(df[c]) else f"NOT bool (dtype={df[c].dtype})"
        else:
            status = "MISSING COLUMN"
        print(f"{df_name}.{c}: {status}")

# Validate that the expected numeric and boolean columns have the correct data types
print("\nEXPECTED DTYPES CHECK:")
check_numeric(menu, expected_numeric_menu, "menu")
check_numeric(sales, expected_numeric_sales, "sales")
check_bool(customers, expected_bool_customers, "customers")

# Identify columns that contain date/time values but are currently strings
print("\nDATE/TIME COLUMNS TO CONVERT:")
print("customers: signup_date")
print("sales: date, time")

# Flag CSV index artifacts if present
if "Unnamed: 0" in sales.columns:
    print("\nNOTE: 'sales' contains an 'Unnamed: 0' column (likely a saved index).")

# Section 3.1: copies, artifact removal, and de-duplication

# Work on copies to preserve original data
menu_clean = menu.copy()
customers_clean = customers.copy()
sales_clean = sales.copy()

# Drop CSV index artifact if present
if "Unnamed: 0" in sales_clean.columns:
    sales_clean = sales_clean.drop(columns=["Unnamed: 0"])
    print("Dropped column 'Unnamed: 0' from sales.")

# Count duplicates before removing them
dupes_before = {
    "menu": menu_clean.duplicated().sum(),
    "customers": customers_clean.duplicated().sum(),
    "sales": sales_clean.duplicated().sum()
}
print("\nDUPLICATES BEFORE CLEANING:", dupes_before)

# Reomve duplicates and reset index
menu_clean = menu_clean.drop_duplicates().reset_index(drop=True)
customers_clean = customers_clean.drop_duplicates().reset_index(drop=True)
sales_clean = sales_clean.drop_duplicates().reset_index(drop=True)

# Condirm duplicates removed
dupes_after = {
    "menu": menu_clean.duplicated().sum(),
    "customers": customers_clean.duplicated().sum(),
    "sales": sales_clean.duplicated().sum()
}
print("DUPLICATES AFTER CLEANING:", dupes_after)

# Section 3.2: numeric type conversion

# Ensure numeric dtypes for price/quantity/discount columns
numeric_map = {
    ("menu", "item_price"): (menu_clean, "item_price"),
    ("sales", "quantity"): (sales_clean, "quantity"),
    ("sales", "unit_price"): (sales_clean, "unit_price"),
    ("sales", "discount"): (sales_clean, "discount"),
    ("sales", "total_price"): (sales_clean, "total_price"),
}

# Convert with coercion to cathch errors
for (df_name, col), (df, col_name) in numeric_map.items():
    before = df[col_name].dtype
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    after = df[col_name].dtype
    if before != after:
        print(f"Coerced {df_name}.{col} -> {after}")

# Report any non-numeric values that became NaN
bad_numeric = {
    "menu.item_price": int(menu_clean["item_price"].isna().sum()),
    "sales.quantity": int(sales_clean["quantity"].isna().sum()),
    "sales.unit_price": int(sales_clean["unit_price"].isna().sum()),
    "sales.discount": int(sales_clean["discount"].isna().sum()),
    "sales.total_price": int(sales_clean["total_price"].isna().sum()),
}
print("Non-numeric (now NaN):", bad_numeric)

# Report any non-numeric values that became NaN
bad_numeric = {
    "menu.item_price": int(menu_clean["item_price"].isna().sum()),
    "sales.quantity": int(sales_clean["quantity"].isna().sum()),
    "sales.unit_price": int(sales_clean["unit_price"].isna().sum()),
    "sales.discount": int(sales_clean["discount"].isna().sum()),
    "sales.total_price": int(sales_clean["total_price"].isna().sum()),
}

print("Non-numeric (now NaN):", bad_numeric)

# Basic range checks
# Quanitites should be >= 0, discounts should be between 0 and 1, prices >= 0
quantity_neg = int((sales_clean["quantity"] < 0).sum())
disc_out = int(((sales_clean["discount"] < 0) | (sales_clean["discount"] > 1)).sum())
price_neg = int(((sales_clean["unit_price"] < 0) | (sales_clean["total_price"] < 0) | (menu_clean["item_price"] < 0)).sum())
print({"quantity_negative": quantity_neg, "discount_out_of_range": disc_out, "price_negative": price_neg})

# Section 3.3: date/time parsing

# Convert customer signup_date to datetime
customers_clean["signup_date"] = pd.to_datetime(customers_clean["signup_date"], errors='coerce')

# Combine sales date and time into a single datetime column
sales_clean["order_datetime"] = pd.to_datetime(
    sales_clean["date"].astype(str) + ' ' + sales_clean["time"].astype(str),
    errors='coerce'
)

# Null check after parsing
print({
    "signup_date_null": int(customers_clean["signup_date"].isna().sum()),
    "order_datetime_null": int(sales_clean["order_datetime"].isna().sum())
})

# Exctract useful time components from order_datetime
sales_clean["order_date"] = sales_clean["order_datetime"].dt.date
sales_clean["order_time"] = sales_clean["order_datetime"].dt.time
sales_clean["order_hour"] = sales_clean["order_datetime"].dt.hour
sales_clean["order_weekday"] = sales_clean["order_datetime"].dt.day_name()

# Sort sales chronologically for reliable time series analysis
sales_clean = sales_clean.sort_values("order_datetime").reset_index(drop=True)

# Sort sales chronologically for reliable time series analysis
sales_clean = sales_clean.sort_values("order_datetime").reset_index(drop=True)

# Preview new columns
print(sales_clean[["order_datetime", "order_date", "order_hour", "order_weekday"]].head())

# Section 3.4: categorical cleanup
import re
def clean_text(val, *, mode = "lower"):
    """Strip spaces, collapse inner whitespace, and convert case."""
    if pd.isna(val):
        return val
    s = re.sub(r'\s+', ' ', str(val).strip())
    if mode == "lower":
        return s.lower()
    if mode == "upper":
        return s.upper()
    if mode == "title":
        return s.title()
    return s

# sales: normalize key text columns
sales_clean["item_name"] = sales_clean["item_name"].apply(lambda x: clean_text(x, mode = "title"))
sales_clean["store_id"] = sales_clean["store_id"].apply(lambda x: clean_text(x, mode = "upper"))
sales_clean["channel"] = sales_clean["channel"].apply(lambda x: clean_text(x, mode = "lower"))
sales_clean["payment_method"] = sales_clean["payment_method"].apply(lambda x: clean_text(x, mode = "lower"))
sales_clean["promo_code"] = sales_clean["promo_code"].apply(lambda x: clean_text(x, mode = "upper"))

# customers: normalize key text columns
customers_clean["home_store"] = customers_clean["home_store"].apply(lambda x: clean_text(x, mode = "upper"))
customers_clean["age_group"] = customers_clean["age_group"].apply(lambda x: clean_text(x, mode = "title"))

# menu: normalize key text columns
menu_clean["item_name"] = menu_clean["item_name"].apply(lambda x: clean_text(x, mode = "title"))
menu_clean["category"] = menu_clean["category"].apply(lambda x: clean_text(x, mode = "title"))

# Constrain to allowed values where applicable
valid_channels = {"in_store", "mobile_app"}
valid_payments = {"card", "cash", "mobile_pay"}

sales_clean["channel"] = sales_clean["channel"].where(sales_clean["channel"].isin(valid_channels), other="in_store")
sales_clean["payment_method"] = sales_clean["payment_method"].where(sales_clean["payment_method"].isin(valid_payments), other="card")

# Data preview
print(sales_clean[["store_id","channel","payment_method","promo_code"]].head())
print(customers_clean[["home_store","age_group"]].head())
print(menu_clean[["item_name","category"]].head())

# Section 3.5: intergirty checks

# Fill missing discount values with 0
sales_clean["discount"] = sales_clean["discount"].fillna(0)

# Recompute expected total from quantity, unit price, and discount
expected_total = (sales_clean["quantity"] * sales_clean["unit_price"] * (1 - sales_clean["discount"])).round(2)

# Compare against recoreded total_price (allowing for small rounding tolerance)
tolerance = 0.01
mismatch_mask = (sales_clean["total_price"] - expected_total).abs() > tolerance

# Report mismatches
mismatch_count = int(mismatch_mask.sum())
print({"total_mismatches": mismatch_count})

# Keep a diffeernce column for auditing
sales_clean["total_diff"] = (sales_clean["total_price"] - expected_total).round(2)

# Optionally fix totals when mismatched
# Uncomment the next line if you want to overwrite with the recomputed value
# sales_clean.loc[mismatch_mask, "total_price"] = expected_total[mismatch_mask]

# Section 3.6: export cleaned datasets

import os

# Write cleaned CSVs next to the script (Section 7 will organize them)
menu_clean.to_csv("cleaned_menu.csv", index = False)
customers_clean.to_csv("cleaned_customers.csv", index = False)
sales_clean.to_csv("cleaned_sales.csv", index = False)

# Confirm saved files
print("\nSaved cleaned files:")
print("  cleaned_menu.csv")
print("  cleaned_customers.csv")
print("  cleaned_sales.csv")



# Section 4.1: top-selling items

# Quantity sold per item
top_quantity = (
    sales_clean.groupby("item_name")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Revenue per item
top_revenue = (
    sales_clean.groupby("item_name")["total_price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Display results
print("\nTOP 10 ITEMS BY QUANTITY SOLD:\n", top_quantity)
print("\n TOP 10 ITEMS BY REVENUE:\n", top_revenue)

# Section 4.2: daily revenue trends

daily_revenue = (
    sales_clean.groupby("order_date")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_date")
)

# Display results
print(daily_revenue.head(10)) # Show first 10 days as a preview
print("\nDAILY REVENUE SHAPE:", daily_revenue.shape)

# Section 4.3: revnenue by hour of day

hourly_revenue = (
    sales_clean.groupby("order_hour")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_hour")
)
print(hourly_revenue)

# Section 4.4: channe;, payment, and loayalty breakdown

# Revenue and order count by channel
channel_summary = (
    sales_clean.groupby("channel") # number of unique orders
    .agg(lines = ("transaction_id", "size"), # total line items
         revenue = ("total_price", "sum") # total revenue
    )
    .reset_index()
)

# Add revenue share for each channel 
channel_summary["revenue_share"] = (channel_summary["revenue"] / channel_summary["revenue"].sum()).round(3)

# Revenue and line counts by payment method
payment_summary = (
    sales_clean.groupby("payment_method")
    .agg(lines = ("transaction_id", "size"), # total line items
         revenue = ("total_price", "sum") # total revenue
    )
    .reset_index()
    .sort_values("revenue", ascending=False)
)

# Calculate the revenue share for each payment method as a percentage of total revenue
# and round the result to 3 decimal places for better readability.
payment_summary["revenue_share"] = (payment_summary["revenue"] / payment_summary["revenue"].sum()).round(3)

# Display the payment summary DataFrame to show the breakdown of revenue by payment method.
print("\nPayment summary:\n", payment_summary)

# Section 5.1: bar chart of top 10 items by revenue

import os
import matplotlib.pyplot as plt

# Create output directory for figures if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Aggregate revune by item and get top 10
top10_revenue = (
    sales_clean.groupby("item_name")["total_price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Plot bar chart
plt.figure()
top10_revenue.plot(kind = "bar")
plt.title("Top 10 Items by Revenue")
plt.xlabel("Item Name")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save and display the figure
plt.savefig("figures/top10_items_revenue.png", dpi = 150)
plt.show()

print("Saved firgure -> figures/top10_items_revenue.png")

# Section 5.2: line chart of daily revenue trend

import matplotlib.pyplot as plt

# Calculate total revenue for each date
daily_revenue = (
    sales_clean.groupby("order_date")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_date")
)

# Creat line chart
plt.figure()
plt.plot(daily_revenue["order_date"], daily_revenue["total_price"], marker='o', linestyle='-')

# Add chart details
plt.title("Daily Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Total Revenue")

# Format x-axis labels for readability
plt.xticks(
    ticks=daily_revenue["order_date"][::3],   # show every 3rd day to reduce clutter
    labels=[d.strftime("%b %d") for d in daily_revenue["order_date"][::3]],  # e.g., Jan 05
    rotation=45,
    ha="right"
)

# Save for reports
plt.savefig("figures/daily_revenue_trend.png", dpi = 150)

plt.show()

# Section 5.3: bar chart of hourly revnue pattern

import matplotlib.pyplot as plt

# Calculate total revenue for each hour of the day
hourly_revenue = (
    sales_clean.groupby("order_hour")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_hour")
)

# Convert numeric hours into AM/PM labels
def format_hour(hour):
    if hour == 0:
        return "12 AM"
    elif hour < 12:
        return f"{hour} AM"
    elif hour == 12:
        return "12 PM"
    else:
        return f"{hour-12} PM"

hour_labels = [format_hour(h) for h in hourly_revenue["order_hour"]]

# Create bar chart
plt.figure()
plt.bar(hourly_revenue["order_hour"], hourly_revenue["total_price"], color='skyblue')

# Add chart details
plt.title("Revene by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Total Revenue")
plt.xticks(hourly_revenue["order_hour"])
plt.tight_layout()

# Save for reports
plt.savefig("figures/hourly_revenue_pattern.png", dpi = 150)

plt.show()

# Section 5.4 bar charts for channel and payment method revenue

import os
import matplotlib.pyplot as plt

# Use FIG_DIR if defined; otherwise save to local 'figures/'
FIG_DIR = "figures"  # Define FIG_DIR if not already defined
fig_out = FIG_DIR
os.makedirs(fig_out, exist_ok = True)

# Calculate revenue by channel 
sales_clean["channel"] = sales_clean["channel"].astype(str).str.strip().str.lower()
expected_channels = ["in_store", "mobile_app"]
channel_revenue = (
    sales_clean.groupby("channel", observed = True)["total_price"]
    .sum()
    .reindex(expected_channels, fill_value = 0.0)
    .reset_index()
)

# Plot revenue by channel
plt.figure()
plt.bar(channel_revenue["channel"], channel_revenue["total_price"], color='lightgreen')
plt.title("Revenue by Channel")
plt.xlabel("Channel")
plt.ylabel("Total Revenue")
plt.tight_layout()

# Save for reports
plt.savefig(os.path.join(fig_out, "revenue_by_channel.png"), dpi = 150)

plt.show()

# Calculate revenue totals by payment method 
sales_clean["payment_method"] = sales_clean["payment_method"].astype(str).str.strip().str.lower()
expected_payments = ["card", "cash", "mobile_pay"]
payment_revenue = (
    sales_clean.groupby("payment_method", observed = True)["total_price"]
    .sum()
    .reindex(expected_payments, fill_value = 0.0)
    .reset_index()
)

# Plot revenue by payment method
plt.figure()
plt.bar(payment_revenue["payment_method"], payment_revenue["total_price"], color='salmon')
plt.title("Revenue by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Total Revenue")
plt.tight_layout()

# Save for reports
plt.savefig(os.path.join(fig_out, "revenue_by_payment_method.png"), dpi = 150)

plt.show()

# Section 5.5: bar chart for loyaty member vs non-member revenue

import matplotlib.pyplot as plt

# Merge sales with customer loyalty status
sales_with_loyalty = sales_clean.merge(
    customers_clean[["customer_id", "is_member"]],
    on="customer_id",
    how="left"
)

# Calculate total revenue by loyalty status
loyalty_revenue = (
    sales_with_loyalty.groupby("is_member", observed = True)["total_price"]
    .sum()
    .reset_index()
    .sort_values("is_member", ascending = False)
)

# Plot revenue by loyalty status
plt.figure()
plt.bar(loyalty_revenue["is_member"].astype(str), loyalty_revenue["total_price"], color='orchid')
plt.title("Revenue by Loyalty Membership")
plt.xlabel("Loyalty Member")
plt.ylabel("Total Revenue")
plt.tight_layout()

# Save for reports
plt.savefig(os.path.join(fig_out, "revenue_by_loyalty_membership.png"), dpi = 150)
plt.close()

plt.show()

# Section 5.6: average order value (AOV) by channel and payment method

import matplotlib.pyplot as plt

# Collapse line items to orders (sum total price per transaction)
orders = (
    sales_clean.groupby("transaction_id")
    .agg(order_total = ("total_price", "sum"),
         customer_id = ("customer_id", "first"))
    .reset_index()
)

# Attach loyalty flag to each order
orders = orders.merge(
    customers_clean[["customer_id", "is_member"]],
    on="customer_id",
    how="left"
)

# Compute AOV and order count by loyalty status
aov_summary = (
    orders.groupby("is_member")
    .agg(orders = ("transaction_id", "count"),
         aov = ("order_total", "mean"),
         revenue = ("order_total", "sum"))
    .reset_index()
    .sort_values("aov", ascending=False)
)

print("\nAOV by Loyalty Membership:\n", aov_summary)

# Plot AOV by loyalty status
plt.figure()
plt.bar(aov_summary["is_member"].astype(str), aov_summary["aov"], color='goldenrod')
plt.title("Average Order Value by Loyalty Status")
plt.xlabel("Loyalty Member")
plt.ylabel("Average Order Value (AOV)")
plt.tight_layout()

# Save for reports
plt.savefig("figures/aov_by_loyalty_status.png", dpi = 150)

plt.show()

# Section 5.7: bar chart for weekday revenue trend

import matplotlib.pyplot as plt

# Calculate revenue totals by day of week
weekday_revenue = (
    sales_clean.groupby("order_weekday")["total_price"]
    .sum()
    .reset_index()
)

# Ensure weekdays are in correct order
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_revenue["order_weekday"] = pd.Categorical(
    weekday_revenue["order_weekday"], categories = weekday_order, ordered = True
)
weekday_revenue = weekday_revenue.sort_values("order_weekday")

# Plot revenue by day of week
plt.figure()
plt.bar(weekday_revenue["order_weekday"], weekday_revenue["total_price"], color='teal')
plt.title("Revenue by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45, ha = "right")
plt.tight_layout()

# Save for reports
plt.savefig("figures/revenue_by_dayofweek.png", dpi = 150)

plt.show()

# Section 6.1: simple sales forecasting with rolling average

import matplotlib.pyplot as plt

# Use the daily revenue data from Section 4.2
daily_revenue = (
    sales_clean.groupby("order_date")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_date")
)

# Add 7-day rolling average column
daily_revenue["7d_ma"] = daily_revenue["total_price"].rolling(window = 7, min_periods = 1).mean()

# Plot average vs rolling average
plt.figure()
plt.plot(daily_revenue["order_date"], daily_revenue["total_price"], label="Actual", marker = "o")
plt.plot(daily_revenue["order_date"], daily_revenue["7d_ma"], label="7-Day Rolling Avg", linewidth=2)

# Add chart details
plt.title("Daily Revenue with 7-Day Rolling Average")
plt.xlabel("Date")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# Save for reports
plt.savefig("figures/daily_revenue_rolling_avg.png", dpi = 150)

plt.show()

# Section 6.2: forcast next 7 days with linear regression

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Prepare daily revenue data
daily_revenue = (
    sales_clean.groupby("order_date")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_date")
)

# Encode dates as numbers (days since start)
daily_revenue["day_num"] = np.arange(len(daily_revenue))

# Train linear regression model
X = daily_revenue[["day_num"]]
y = daily_revenue["total_price"]
model = LinearRegression()
model.fit(X, y)

# Predict next 7 days
future_days = np.arange(len(daily_revenue), len(daily_revenue) + 7).reshape(-1, 1)
future_preds = model.predict(future_days)

# Build forecast DataFrame
forecast = pd.DataFrame({
    "order_date": pd.date_range(start = daily_revenue["order_date"].iloc[-1] + pd.Timedelta(days = 1), periods = 7),
    "predicted_revenue": future_preds
})

# Plot actual and forecasted revenue
plt.figure()
plt.plot(daily_revenue["order_date"], daily_revenue["total_price"], label="Actual", marker='o')
plt.plot(daily_revenue["order_date"], model.predict(X), label = "Trend Line", linewidth = 2)
plt.plot(forecast["order_date"], forecast["predicted_revenue"], label = "Forecast (7 days)", linestyle = "--", marker = "o")

# Add chart details
plt.title("Daily Revenue with Forecast (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# Save for reports
plt.savefig("figures/daily_revenue_forecast.png", dpi = 150)

plt.show()

# Display forecasted values
print("\n7-Day Revenue Forecast:\n", forecast)

# Section 6.3: forecast using weekday effects (captures day-of-week patterns)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Aggregate total revenue per day and add dummy data for weekdays
daily_revenue = (
    sales_clean.groupby("order_date")["total_price"]
    .sum()
    .reset_index()
    .sort_values("order_date")
    .rename(columns={"total_price": "revenue"})
)

# Add trend and weekday features
daily_revenue["day_num"] = np.arange(len(daily_revenue)) # numeric index of days
daily_revenue["weekday"] = pd.to_datetime(daily_revenue["order_date"]).dt.day_name() # extract weekday name
X = pd.get_dummies(daily_revenue[["day_num", "weekday"]], drop_first=True)
y = daily_revenue["revenue"]

# Train linear regression on trend and weekday features
lr = LinearRegression()
lr.fit(X, y)

# Create next 7 dates with matching features
start_next = daily_revenue["order_date"].iloc[-1] + pd.Timedelta(days = 1)  # day after last actual date
future_dates = pd.date_range(start = start_next, periods = 7, freq = "D")   # next 7 calendar days
future = pd.DataFrame({"order_date": future_dates})                         # build future DataFrame

# Add numeric day index and weekday name for future dates
future["day_num"] = np.arange(daily_revenue["day_num"].iloc[-1] + 1,
                              daily_revenue["day_num"].iloc[-1] + 1 + len(future))
future["weekday"] = future["order_date"].dt.day_name()

# Encode weekday for future dates
future_X = pd.get_dummies(future[["day_num", "weekday"]], drop_first = True)

# Add any missing weekday columns so structure matches training
for col in X.columns:
    if col not in future_X.columns:
        future_X[col] = 0

# Reorder columns to match training input
future_X = future_X[X.columns]

# Predict revenue for future dates
future["predicted_revenue"] = lr.predict(future_X)

# Plot actual, fitted line, and forecast
plt.figure()
plt.plot(daily_revenue["order_date"], y, label = "Actual", marker = "o", linewidth = 1)
plt.plot(daily_revenue["order_date"], lr.predict(X), label = "Fitted", linewidth = 2)
plt.plot(future["order_date"], future["predicted_revenue"], label = "Forecast", linestyle = "--", marker = "o")
plt.title("Daily Revenue Forecast with Weekday Effects")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.xticks(rotation = 45, ha = "right")
plt.legend()
plt.tight_layout()

# Save for reports
plt.savefig("figures/daily_revenue_forecast_weekday.png", dpi = 150)

plt.show()

# Display forecasted values for next 7 days
print("\nForecast (next 7 days):\n", future[["order_date", "predicted_revenue"]])

# Section 6.4: customer segmentation (RFM analysis)

import pandas as pd

# Collapse transactions to costomer level with last purchase, order count, and total spend
rfm = (
    sales_clean.groupby("customer_id")
    .agg(
        last_purchase = ("order_date", "max"),
        order_count = ("transaction_id", "nunique"),
        total_spend = ("total_price", "sum")
    )
    .reset_index()
)

# Calculate recency in days from last purchase to most recent date in dataset
most_recent_date = pd.to_datetime(sales_clean["order_date"]).max() + pd.Timedelta(days = 1)
rfm["recency_days"] = (most_recent_date - pd.to_datetime(rfm["last_purchase"])).dt.days


# Display RFM summary
print("\nRFM Summary:\n", rfm.head())

# Section 6.5: RFM scoring

import pandas as pd

# Rank values to avoid issues with duplicate bin edges
ranked_recency = rfm["recency_days"].rank(method = "first")
ranked_orders  = rfm["order_count"].rank(method = "first")
ranked_spend   = rfm["total_spend"].rank(method = "first")

# Assign quartile-based scores (1 = low, 4 = high)
# Recency: fewer days = higher score
rfm["recency_score"] = pd.qcut(ranked_recency, q = 4, labels = [4, 3, 2, 1]).astype(int)
# Order count: more orders = higher score
rfm["frequency_score"] = pd.qcut(ranked_orders, q = 4, labels = [1, 2, 3, 4]).astype(int)
# Spend: more spending = higher score
rfm["monetary_score"] = pd.qcut(ranked_spend, q = 4, labels = [1, 2, 3, 4]).astype(int)

# Combine RFM scores
rfm["rfm_total"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

# Assign simple segment labels based on total score
def label_segment(row):
    if row["rfm_total"] >= 10:
        return "Top"
    if row["rfm_total"] >= 7:
        return "Loyal"
    if row["rfm_total"] >= 5:
        return "Promising"
    return "At Risk"

# Apply segment labeling

rfm["segment"] = rfm.apply(label_segment, axis = 1)

# Show preview with scores and segment
print("\nRFM Scores:\n", rfm[[
    "customer_id", "recency_days", "order_count", "total_spend",
    "recency_score", "frequency_score", "monetary_score", "rfm_total", "segment"
]].head())

# Section 6.6: segment summary

# Count customers and revenue by segment
segment_summary = (
    rfm.groupby("segment")
    .agg(
        customer_count = ("customer_id", "nunique"),
        total_revenue = ("total_spend", "sum")
    )
    .reset_index()
    .sort_values("total_revenue", ascending = False)
)

# Add revenue share
segment_summary["revenue_share"] = (
    segment_summary["total_revenue"] / segment_summary["total_revenue"].sum()
).round(3)

# Display segment summary

print("\nSegment Summary:\n", segment_summary)

# Section 6.7: segment visualization

import matplotlib.pyplot as plt

# Bar chart: customer count by segment
plt.figure()
plt.bar(segment_summary["segment"], segment_summary["customer_count"])
plt.title("Customer Count by Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()

# Save for reports
plt.savefig("figures/segment_customer_count.png", dpi = 150)

# Display the figure
plt.show()

# Bar chart: revenue share by segment
plt.figure()
plt.bar(segment_summary["segment"], segment_summary["revenue_share"])
plt.title("Revenue Share by Segment")
plt.xlabel("Segment")
plt.ylabel("Revenue Share")
plt.tight_layout()

# Save for reports
plt.savefig("figures/segment_revenue_share.png", dpi = 150)

# Display the figure
plt.show()

# Section 7: organize outputs

import os
import shutil
from glob import glob

# Create a dedicated folder for this run's outputs
output_dir = "starbucks_sales_analytics_outputs"
os.makedirs(output_dir, exist_ok = True)

# Known files that may have been created directly in the project root
root_files = [
    "cleaned_menu.csv",
    "cleaned_customers.csv",
    "cleaned_sales.csv",
    "top10_items_revenue.png",
    "daily_revenue_trend.png",
    "hourly_revenue.png",
    "revenue_by_channel.png",
    "revenue_by_payment.png",
    "revenue_by_loyalty.png",
    "aov_by_loyalty.png",
    "revenue_by_weekday.png",
    "daily_revenue_forecast_lr.png",
    "daily_revenue_forecast_weekday.png",
    "segment_customer_count.png",
    "segment_revenue_share.png",
]

# Also collect any figures saved under a 'figures/' folder (if used)
figure_dir = "figures"
figure_files = glob(os.path.join(figure_dir, "*.png")) if os.path.isdir(figure_dir) else []

# Move root-level created files (only if they exist)
for f in root_files:
    if os.path.exists(f):
        shutil.move(f, os.path.join(output_dir, os.path.basename(f)))

# Move figures (if any) and clean up the empty 'figures/' folder
for f in figure_files:
    shutil.move(f, os.path.join(output_dir, os.path.basename(f)))
if figure_files and not os.listdir(figure_dir):
    os.rmdir(figure_dir)

# Final confirmation
print(f"\nAll created files have been organized into: {output_dir}/")
