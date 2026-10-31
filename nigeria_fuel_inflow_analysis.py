# ==========================================================
# NIGERIA FUEL INFLOW PREDICTION AND COMPARATIVE ANALYSIS
# Comparing 2022 with 2026
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ----------------------------------------------------------
# 1. HISTORICAL BENCHMARK
# ----------------------------------------------------------

baseline_2022 = 66.7  # Million litres per day

# Monthly reported average supply in 2026
# Values are in million litres per day

data = {
    "Month": [
        "January", "February", "March",
        "April", "May", "June",
        "July", "August", "September"
    ],

    "Domestic_Supply": [
        40.1, 29.4, 34.2, 40.7, 41.5,
        32.5, 25.8, 35.9, 28.5
    ],

    "Imported_Supply": [
        24.8, 3.0, 5.9, 3.7, 5.9,
        18.1, 19.7, 14.6, 10.2
    ]
}

df = pd.DataFrame(data)

# ----------------------------------------------------------
# 2. CALCULATE TOTAL FUEL INFLOW
# ----------------------------------------------------------

df["Total_Supply"] = (
    df["Domestic_Supply"] +
    df["Imported_Supply"]
)

print("\nNIGERIA FUEL INFLOW REPORT")
print("=" * 60)

print(df.to_string(index=False))

# ----------------------------------------------------------
# 3. COMPARE 2026 WITH 2022
# ----------------------------------------------------------

latest_supply = df["Total_Supply"].iloc[-1]

difference = latest_supply - baseline_2022

percentage_change = (
    difference / baseline_2022
) * 100

print("\n2022 VS 2026 COMPARISON")
print("=" * 60)

print(f"2022 Daily Supply: {baseline_2022:.2f} Million Litres")

print(f"September 2026 Supply: {latest_supply:.2f} Million Litres")

print(f"Supply Difference: {difference:.2f} Million Litres")

print(f"Percentage Change: {percentage_change:.2f}%")

if difference > 0:
    print("Result: Fuel supply has increased.")
elif difference < 0:
    print("Result: Fuel supply has decreased.")
else:
    print("Result: Fuel supply remains unchanged.")

# ----------------------------------------------------------
# 4. MACHINE LEARNING PREDICTION
# ----------------------------------------------------------

# Convert months into numerical values
X = np.array(range(1, 10)).reshape(-1, 1)

y = df["Total_Supply"].values

# Train Linear Regression Model
model = LinearRegression()

model.fit(X, y)

# Predict October, November and December
future_months = np.array([10, 11, 12]).reshape(-1, 1)

predictions = model.predict(future_months)

# Prevent negative supply predictions
predictions = np.maximum(predictions, 0)

future_data = pd.DataFrame({
    "Month": ["October", "November", "December"],
    "Predicted_Supply": predictions
})

print("\nFUTURE FUEL INFLOW PREDICTION")
print("=" * 60)

print(future_data.to_string(index=False))

# ----------------------------------------------------------
# 5. MODEL PERFORMANCE
# ----------------------------------------------------------

training_predictions = model.predict(X)

mae = mean_absolute_error(y, training_predictions)

print("\nMODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error: {mae:.2f} Million Litres/Day")

print("Model Used: Linear Regression")

# ----------------------------------------------------------
# 6. FUEL SUPPLY ANALYSIS
# ----------------------------------------------------------

average_supply = df["Total_Supply"].mean()

average_domestic = df["Domestic_Supply"].mean()

average_imported = df["Imported_Supply"].mean()

print("\nSUPPLY ANALYSIS")
print("=" * 60)

print(f"Average Total Supply: {average_supply:.2f} ML/Day")

print(f"Average Domestic Supply: {average_domestic:.2f} ML/Day")

print(f"Average Imported Supply: {average_imported:.2f} ML/Day")

# ----------------------------------------------------------
# 7. GRAPHICAL COMPARISON
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    df["Month"],
    df["Total_Supply"],
    marker="o",
    label="2026 Actual Supply"
)

plt.axhline(
    y=baseline_2022,
    linestyle="--",
    label="2022 Average Supply"
)

plt.plot(
    ["October", "November", "December"],
    predictions,
    marker="s",
    linestyle=":",
    label="2026 Predicted Supply"
)

plt.title("Nigeria Fuel Inflow: 2022 vs 2026 Prediction")

plt.xlabel("Months")

plt.ylabel("Million Litres Per Day")

plt.xticks(rotation=45)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# 8. DOMESTIC VS IMPORTED FUEL SUPPLY
# ----------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    df["Month"],
    df["Domestic_Supply"],
    label="Domestic Supply"
)

plt.bar(
    df["Month"],
    df["Imported_Supply"],
    bottom=df["Domestic_Supply"],
    label="Imported Supply"
)

plt.title("Nigeria Domestic and Imported Petrol Supply")

plt.xlabel("Months")

plt.ylabel("Million Litres Per Day")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# 9. FINAL PROJECT CONCLUSION
# ----------------------------------------------------------

print("\nFINAL PROJECT CONCLUSION")
print("=" * 60)

if latest_supply < baseline_2022:
    print(
        "Nigeria's September 2026 petrol supply "
        "was below the 2022 daily supply benchmark."
    )
else:
    print(
        "Nigeria's September 2026 petrol supply "
        "was above the 2022 daily supply benchmark."
    )

print(
    "\nThe machine learning model has generated "
    "a three-month forecast based on the 2026 trend."
)

print("\nEND OF ANALYSIS")
