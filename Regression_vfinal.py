import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load dataset
df = pd.read_excel('Final dataset.xlsx')

# Constants
share_debt = 0.7  # Proportion of debt in financing structure

# Define regions and technologies
regions = ['Europe & Central Asia', 'East Asia & Pacific', 'Middle East & North Africa',
           'North America', 'South Asia', 'Sub-Saharan Africa', 'Latin America & Caribbean']
technologies = ['Solar PV', 'Wind onshore', 'Wind offshore']

# Set reference categories
ref_region = "Europe & Central Asia"  # Reference category for region
ref_tech = "Solar PV"  # Reference category for technology

# Create dummy variables for all regions and technologies (no reference category is dropped)
for region in regions:
    df[f'PR_{region}'] = (df['Region'] == region).astype(int)

for tech in technologies:
    df[f'PT_{tech}'] = (df['Technology'] == tech).astype(int)

# Extract relevant columns
T = df['Tax Rate']
RFR = df['Global Risk-Free Rate']
S = df['Country Default Spread']
P_E = df['Equity Risk Premium']
P_C = df['Country Premium']
act_WACC = df['WACC (nominal, after-tax)']

# Calculate Cost of Debt and Cost of Equity without technology and region premia
cost_debt = (RFR + S + 0.02) * (1 - T) # 0.02 = Infrastructure premium
cost_equity = RFR + P_E + P_C

# Calculate base WACC without premiums
base_WACC = share_debt * cost_debt + (1 - share_debt) * cost_equity

# Evaluate performance before regression
original_r_value = np.corrcoef(base_WACC, act_WACC)[0, 1]
original_r_squared = original_r_value ** 2
original_mape = np.mean(np.abs((act_WACC - base_WACC) / act_WACC)) * 100
original_aae = mean_absolute_error(act_WACC, base_WACC)

print("Performance before regression:")
print(f"R-value: {original_r_value}")
print(f"R-squared: {original_r_squared}")
print(f"MAPE: {original_mape:.2f}%")
print(f"AAE: {original_aae:.4f}")

# Adjust WACC to account for the premiums
df['Adjusted_WACC'] = act_WACC - base_WACC

## Extract relevant columns (Keep all region & technology dummies for regression through the origin)
region_vars = [f'PR_{region}' for region in regions]  # No category is dropped
tech_vars = [f'PT_{tech}' for tech in technologies]  # No category is dropped

# Prepare feature matrix (X) and target variable (y)
X_full = df[region_vars + tech_vars]  # For Regression through the Origin
X_ols = sm.add_constant(df[[f'PR_{region}' for region in regions if region != ref_region] +
                           [f'PT_{tech}' for tech in technologies if tech != ref_tech]])

# Target variable
y = df['Adjusted_WACC']

# Technology premiums are properly scaled and converted to float
X_full = X_full.copy()
X_full[tech_vars] = X_full[tech_vars].astype(float).mul(1 - share_debt)

# Exclude 'PT_Solar PV' from OLS adjustments because it does not exist in X_ols
ols_tech_vars = [f'PT_{tech}' for tech in technologies if tech != ref_tech]

X_ols = X_ols.copy()  # X_ols as new DataFrame copy for Standard OLS
X_ols[ols_tech_vars] = X_ols[ols_tech_vars].astype(float).mul(1 - share_debt)

# Add constant term for OLS
X_ols = sm.add_constant(X_ols)

# Fit OLS regression model
model_ols = sm.OLS(y, X_ols).fit()

# Print model summary
print("\n==== OLS Regression Results (With Intercept) ====")
print(model_ols.summary())

# Extract coefficients
coefficients = model_ols.params
print("Coefficients:")
print(coefficients)

# Calculate predicted WACC including regression premiums
predicted_WACC = base_WACC + model_ols.predict(X_ols)

# Evaluate performance after regression
r_value_ols = np.corrcoef(predicted_WACC, act_WACC)[0, 1]
r_squared_ols = r_value_ols ** 2
mape_ols = np.mean(np.abs((act_WACC - predicted_WACC) / act_WACC)) * 100
aae_ols = mean_absolute_error(act_WACC, predicted_WACC)

print("\nPerformance after regression:")
print(f"R-value: {r_value_ols}")
print(f"R-squared: {r_squared_ols}")
print(f"MAPE: {mape_ols:.2f}%")
print(f"AAE: {aae_ols:.4f}")

# Define subsets for analysis
regions_subset = regions
technologies_subset = technologies
time_bins = {
    "Pre-2016": df['Financing year'] < 2016,
    "2016-2019": (df['Financing year'] >= 2016) & (df['Financing year'] <= 2019),
    "2020-2023": df['Financing year'] >= 2020
}

# Store results
subset_results = []

# Evaluate performance on each region
for region in regions_subset:
    subset_idx = df['Region'] == region
    if subset_idx.sum() == 0:
        continue
    preds = base_WACC[subset_idx] + model_ols.predict(X_ols[subset_idx])
    actual = act_WACC[subset_idx]

    r_val = np.corrcoef(preds, actual)[0, 1]
    r2 = r_val ** 2
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    aae = mean_absolute_error(actual, preds)

    subset_results.append(["Region", region, r_val, r2, mape, aae])

# Evaluate performance on each technology
for tech in technologies_subset:
    subset_idx = df['Technology'] == tech
    if subset_idx.sum() == 0:
        continue
    preds = base_WACC[subset_idx] + model_ols.predict(X_ols[subset_idx])
    actual = act_WACC[subset_idx]

    r_val = np.corrcoef(preds, actual)[0, 1]
    r2 = r_val ** 2
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    aae = mean_absolute_error(actual, preds)

    subset_results.append(["Technology", tech, r_val, r2, mape, aae])

# Evaluate performance across time periods
for label, condition in time_bins.items():
    subset_idx = condition
    if subset_idx.sum() == 0:
        continue
    preds = base_WACC[subset_idx] + model_ols.predict(X_ols[subset_idx])
    actual = act_WACC[subset_idx]

    r_val = np.corrcoef(preds, actual)[0, 1]
    r2 = r_val ** 2
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    aae = mean_absolute_error(actual, preds)

    subset_results.append(["Time Period", label, r_val, r2, mape, aae])

# Convert to DataFrame for reporting
subset_perf_df = pd.DataFrame(subset_results, columns=["Subset Type", "Subset", "R-value", "R²", "MAPE (%)", "AAE"])
subset_perf_df.to_csv("subset_performance_ols.csv", index=False)

# Print results
from tabulate import tabulate
print("\nOLS Regression Performance by Subset:")
print(tabulate(subset_perf_df, headers="keys", tablefmt="pretty"))

# Compare results
print("\nComparison of results:")
print(f"R-squared improvement: {r_squared_ols - original_r_squared}")
print(f"MAPE improvement: {original_mape - mape_ols:.2f}%")
print(f"AAE improvement: {original_aae - aae_ols:.4f}")

### **VARIANCE INFLATION FACTOR (VIF) CHECK**
# Compute VIF using X_ols (excluding the constant term)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_ols.columns
vif_data["VIF"] = [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])]

print("\nVIF Results:")
print(vif_data)

### **2. REGRESSION THROUGH THE ORIGIN (NO INTERCEPT)**
# Fit regression through the origin (no intercept, using all categories)
model_origin = sm.OLS(y, X_full).fit()

print("\n==== OLS Regression Results (Through the Origin) ====")
print(model_origin.summary())

# Extract coefficients
coefficients = model_origin.params
print("Coefficients:")
print(coefficients)

# Predict using regression through the origin
predicted_WACC_origin = base_WACC + model_origin.predict(X_full)

# Evaluate performance after regression through the origin
r_value_origin = np.corrcoef(predicted_WACC_origin, act_WACC)[0, 1]
r_squared_origin = r_value_origin ** 2
mape_origin = np.mean(np.abs((act_WACC - predicted_WACC_origin) / act_WACC)) * 100
aae_origin = mean_absolute_error(act_WACC, predicted_WACC_origin)

print("\nPerformance after regression through the origin:")
print(f"R-value: {r_value_origin}")
print(f"R-squared: {r_squared_origin}")
print(f"MAPE: {mape_origin:.2f}%")
print(f"AAE: {aae_origin:.4f}")


