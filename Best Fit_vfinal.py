#import libraries and define variables
import numpy as np
import pandas as pd

# Read the Excel file
df = pd.read_excel('Final dataset.xlsx')

# Get recorded WACC values
act_WACC = df['WACC (nominal, after-tax)'].values.tolist()

print("Average WACC in dataset: ",np.average(act_WACC))
print("Number of data points: ", len(act_WACC))


#%% CALCULATION OF REGION PREMIUM
df = pd.read_excel('Data compilation_IRENA components_vfinal_CPI.xlsx')

act_WACC = df['WACC (nominal, after-tax)'].values.tolist()

count1 = df[(df['Region'] == 'Europe & Central Asia')].shape[0]
count2 = df[(df['Region'] == 'East Asia & Pacific')].shape[0]
count3 = df[(df['Region'] == 'Middle East & North Africa')].shape[0]
count4 = df[(df['Region'] == 'North America')].shape[0]
count5 = df[(df['Region'] == 'South Asia')].shape[0]
count6 = df[(df['Region'] == 'Sub-Saharan Africa')].shape[0]
count7 = df[(df['Region'] == 'Latin America & Caribbean')].shape[0]
print("Number of WACCs per Region: Europe & Central Asia: " ,count1 ,"// East Asia & Pacific: " ,count2
      ,"// Middle East & North Africa: " ,count3 ,"// North America: " ,count4 ,"// South Asia: " ,count5
      ,"// Sub-Saharan Africa: ", count6, "// Latin America & Caribbean: " ,count7)

# Started from large ranges down to medium, then smallest ranges to identify # the best region premium values for each region with decrease in interval size and stepsize
# largest ranges (Interval size <0.1, stepsize 0.01)
# Leave only one section outcommented at once (large, middle, smallest)
'''PR_ranges = {
    'Europe & Central Asia': np.arange(0, 0.06, 0.01), 
    'East Asia & Pacific': np.arange(0, 0.06, 0.01),
    'Middle East & North Africa': np.arange(0, 0.06, 0.01),
    'North America': np.arange(0, 0.06, 0.01),
    'South Asia': np.arange(0.0, 0.06, 0.01),
    'Sub-Saharan Africa': np.arange(0.0, 0.06, 0.01),
    'Latin America & Caribbean': np.arange(0.0, 0.06, 0.01)
}'''

#middle sized ranges (Interval size <0.05, stepsize 0.005)
'''PR_ranges = {
    'Europe & Central Asia': np.arange(0.00, 0.03, 0.005),
    'East Asia & Pacific': np.arange(0.01, 0.04, 0.005),
    'Middle East & North Africa': np.arange(0.01, 0.04, 0.005),
    'North America': np.arange(0.01, 0.04, 0.005),
    'South Asia': np.arange(0.03, 0.055, 0.005),
    'Sub-Saharan Africa': np.arange(0.04, 0.06, 0.005),
    'Latin America & Caribbean': np.arange(0.02, 0.05, 0.005)
}'''


# smallest ranges (Interval size <=0.01, stepsize 0.001)
PR_ranges = {
    'Europe & Central Asia': np.arange(0.013, 0.019, 0.001),
    'East Asia & Pacific': np.arange(0.023, 0.027, 0.001),
    'Middle East & North Africa': np.arange(0.023, 0.027, 0.001),
    'North America': np.arange(0.023, 0.027, 0.001),
    'South Asia': np.arange(0.046, 0.052, 0.001),
    'Sub-Saharan Africa': np.arange(0.054, 0.064, 0.001),
    'Latin America & Caribbean': np.arange(0.034, 0.037, 0.001)
}

# Extract the necessary columns from the dataframe
T = df['Tax Rate'].values.tolist()
RFR = df['Global Risk-Free Rate'].values.tolist()
S = df['Country Default Spread'].values.tolist()
P_E = df['Equity Risk Premium'].values.tolist()
P_C = df['Country Premium'].values.tolist()
Region = df['Region'].values.tolist()
share_debt = 0.7

# Initialize variables to track the best combination
best_combination = {}
lowest_error = float('inf')
highest_r_value = float('-inf')

# Function to calculate r-value (example using a placeholder function)
def calculate_r_value(predicted, actual):
    correlation_matrix = np.corrcoef(predicted, actual)
    r_value = correlation_matrix[0, 1]
    return r_value

# 7 nested loops
for PR_EUR in PR_ranges['Europe & Central Asia']:
    for PR_EAP in PR_ranges['East Asia & Pacific']:
        for PR_MENA in PR_ranges['Middle East & North Africa']:
            for PR_NA in PR_ranges['North America']:
                for PR_SA in PR_ranges['South Asia']:
                    for PR_SSA in PR_ranges['Sub-Saharan Africa']:
                        for PR_LAC in PR_ranges['Latin America & Caribbean']:

                            B_WACC = []
                            perc_error = []

                            for i in range(len(T)):
                                reg = Region[i]

                                if reg == 'Europe & Central Asia':
                                    PR = PR_EUR
                                elif reg == 'East Asia & Pacific':
                                    PR = PR_EAP
                                elif reg == 'Middle East & North Africa':
                                    PR = PR_MENA
                                elif reg == 'North America':
                                    PR = PR_NA
                                elif reg == 'South Asia':
                                    PR = PR_SA
                                elif reg == 'Sub-Saharan Africa':
                                    PR = PR_SSA
                                elif reg == 'Latin America & Caribbean':
                                    PR = PR_LAC
                                else:
                                    continue  # If the technology is not recognized, skip this iteration

                                cost_debt = (RFR[i] + S[i] + 0.02 +PR) * (1 - T[i]) #0.02 = static infrastructure premium, no technology premium on cost of debt (as defined in specifications A-D)
                                cost_equity = RFR[i] + P_E[i] + P_C[i] + 0.02 + PR #0.02 = static technology premium

                                # Calculate WACC and append to B_WACC
                                B_WACC.append(cost_debt * share_debt + cost_equity * (1 - share_debt))

                                # Calculate absolute % error
                                perc_error.append(np.abs((act_WACC[i] - B_WACC[i]) / act_WACC[i]))

                            # Calculate average percentage error and r-value
                            avg_perc_error = np.mean(perc_error)
                            r_value = calculate_r_value(B_WACC, act_WACC)

                            # Update the best combination for lowest error
                            if avg_perc_error < lowest_error:
                                lowest_error = avg_perc_error
                                best_combination['lowest_error'] = {
                                    'EUR': PR_EUR,
                                    'EAP': PR_EAP,
                                    'MENA': PR_MENA,
                                    'NA': PR_NA,
                                    'SA': PR_SA,
                                    'SSA': PR_SSA,
                                    'LAC': PR_LAC,
                                    'error': avg_perc_error
                                }

                            # Update the best combination for highest r-value
                            if r_value > highest_r_value:
                                highest_r_value = r_value
                                best_combination['highest_r_value'] = {
                                    'EUR': PR_EUR,
                                    'EAP': PR_EAP,
                                    'MENA': PR_MENA,
                                    'NA': PR_NA,
                                    'SA': PR_SA,
                                    'SSA': PR_SSA,
                                    'LAC': PR_LAC,
                                    'r_value': r_value
                                }

# Output the best combinations
print("Best combination for highest r-value:", best_combination['highest_r_value'])
print("Best combination for lowest mean average percentage error:", best_combination['lowest_error']) #robustness check on similarity of region premium coefficients

#Performance analysis
best = best_combination['highest_r_value']

PR_values = {
    'Europe & Central Asia': best['EUR'],
    'East Asia & Pacific': best['EAP'],
    'Middle East & North Africa': best['MENA'],
    'North America': best['NA'],
    'South Asia': best['SA'],
    'Sub-Saharan Africa': best['SSA'],
    'Latin America & Caribbean': best['LAC']
}

B_WACC_best = []
abs_perc_errors = []
abs_errors = []

for i in range(len(T)):
    reg = Region[i]
    PR = PR_values.get(reg, 0)

    cost_debt = (RFR[i] + S[i] + 0.02 + PR) * (1 - T[i])
    cost_equity = RFR[i] + P_E[i] + P_C[i] + 0.02 + PR
    pred_wacc = cost_debt * share_debt + cost_equity * (1 - share_debt)

    B_WACC_best.append(pred_wacc)
    abs_perc_errors.append(abs((act_WACC[i] - pred_wacc) / act_WACC[i]))
    abs_errors.append(abs(act_WACC[i] - pred_wacc))

# Compute metrics
r_value_final = calculate_r_value(B_WACC_best, act_WACC)
mape = np.mean(abs_perc_errors) * 100  # as a percentage
aae = np.mean(abs_errors)

# Print metrics
print("\nEvaluation for Best Combination (Highest R-value):")
print("R-value:", round(r_value_final, 4))
print("MAPE (%):", round(mape, 2))
print("AAE:", round(aae, 4))



#%% CALCULATION OF TECHNOLOGY PREMIUM

# Started from large ranges down to medium, then smallest ranges to identify the best TECHNOLOGY premium values for each TECHNOLOGY with decrease in interval size and stepsize

# largest ranges (Interval size <0.1, stepsize 0.01)
'''P_T_ranges = {
    'Solar PV': np.arange(0.00, 0.1, 0.01),
    'Wind onshore': np.arange(0.00, 0.1, 0.01),
    'Wind offshore': np.arange(0.00, 0.1, 0.01)
}'''

#middle sized ranges (Interval size <0.05, stepsize 0.005)
'''P_T_ranges = {
    'Solar PV': np.arange(0.00, 0.05, 0.005),
    'Wind onshore': np.arange(0.00, 0.05, 0.005),
    'Wind offshore': np.arange(0.02, 0.07, 0.005)
}'''

# smallest ranges (Interval size <=0.01, stepsize 0.001)
P_T_ranges = {
    'Solar PV': np.arange(0.005, 0.015, 0.001),
    'Wind onshore': np.arange(0.00, 0.01, 0.001),
    'Wind offshore': np.arange(0.035, 0.045, 0.001)
}

# Extract the necessary columns from the dataframe
T = df['Tax Rate'].values.tolist()
RFR = df['Global Risk-Free Rate'].values.tolist()
S = df['Country Default Spread'].values.tolist()
P_E = df['Equity Risk Premium'].values.tolist()
P_C = df['Country Premium'].values.tolist()
Technology = df['Technology'].values.tolist()
reg = df['Region'].values.tolist()

share_debt = 0.7

# Initialize variables to track the best combination
best_combination = {}
lowest_avg_abs_diff = float('inf')
highest_r_value = float('-inf')


# Function to calculate r-value (example using a placeholder function)
def calculate_r_value(predicted, actual):
    correlation_matrix = np.corrcoef(predicted, actual)
    r_value = correlation_matrix[0, 1]
    return r_value

# Triple nested loop
for P_T_solar in P_T_ranges['Solar PV']:
    for P_T_wind_onshore in P_T_ranges['Wind onshore']:
        for P_T_wind_offshore in P_T_ranges['Wind offshore']:

            B_WACC = []
            abs_diff = []

            for i in range(len(T)):
                tech = Technology[i]

                if reg[i] == 'Europe & Central Asia':
                    PR = 0.015
                elif reg[i] == 'East Asia & Pacific':
                    PR = 0.026
                elif reg[i] == 'Middle East & North Africa':
                    PR = 0.025
                elif reg[i] == 'North America':
                    PR = 0.026
                elif reg[i] == 'South Asia':
                    PR = 0.050
                elif reg[i] == 'Sub-Saharan Africa':
                    PR = 0.054
                elif reg[i] == 'Latin America & Caribbean':
                    PR = 0.034

                if tech == 'Solar PV':
                    P_T = P_T_solar
                elif tech == 'Wind onshore':
                    P_T = P_T_wind_onshore
                elif tech == 'Wind offshore':
                    P_T = P_T_wind_offshore
                else:
                    continue  # If the technology is not recognized, skip this iteration

                cost_debt = (RFR[i] + S[i] + 0.02 + PR) * (1 - T[i]) #0.02 = static infrastructure premium, no technology premium on cost of debt (as defined in specifications A-D)
                cost_equity = RFR[i] + P_E[i] + P_C[i] + P_T + PR

                # Calculate WACC and append to B_WACC
                B_WACC.append(cost_debt * share_debt + cost_equity * (1 - share_debt))

                # Calculate absolute difference
                abs_diff.append(np.abs(act_WACC[i] - B_WACC[i]))

            # Calculate average absolute difference and r-value
            avg_abs_diff = np.mean(abs_diff)
            r_value = calculate_r_value(B_WACC, act_WACC)

            # Update the best combination for lowest average absolute difference
            if avg_abs_diff < lowest_avg_abs_diff:
                lowest_avg_abs_diff = avg_abs_diff
                best_combination['lowest_avg_abs_diff'] = {
                    'P_T_solar': P_T_solar,
                    'P_T_wind_onshore': P_T_wind_onshore,
                    'P_T_wind_offshore': P_T_wind_offshore,
                    'avg_abs_diff': avg_abs_diff
                }

            # Update the best combination for highest r-value
            if r_value > highest_r_value:
                highest_r_value = r_value
                best_combination['highest_r_value'] = {
                    'P_T_solar': P_T_solar,
                    'P_T_wind_onshore': P_T_wind_onshore,
                    'P_T_wind_offshore': P_T_wind_offshore,
                    'r_value': r_value
                }

# Output the best combinations
print("Best combination for highest r-value:", best_combination['highest_r_value'])

#Performance analysis

best = best_combination['highest_r_value']

P_T_values = {
    'Solar PV': best['P_T_solar'],
    'Wind onshore': best['P_T_wind_onshore'],
    'Wind offshore': best['P_T_wind_offshore']
}

B_WACC_best = []
abs_perc_errors = []
abs_errors = []

for i in range(len(T)):
    tech = Technology[i]

    # Fixed region premiums (used during optimization)
    if reg[i] == 'Europe & Central Asia':
        PR = 0.015
    elif reg[i] == 'East Asia & Pacific':
        PR = 0.026
    elif reg[i] == 'Middle East & North Africa':
        PR = 0.025
    elif reg[i] == 'North America':
        PR = 0.026
    elif reg[i] == 'South Asia':
        PR = 0.050
    elif reg[i] == 'Sub-Saharan Africa':
        PR = 0.054
    elif reg[i] == 'Latin America & Caribbean':
        PR = 0.034
    else:
        continue

    P_T = P_T_values.get(tech, 0)

    cost_debt = (RFR[i] + S[i] + 0.02 + PR) * (1 - T[i])
    cost_equity = RFR[i] + P_E[i] + P_C[i] + P_T + PR
    pred_wacc = cost_debt * share_debt + cost_equity * (1 - share_debt)

    B_WACC_best.append(pred_wacc)
    abs_perc_errors.append(abs((act_WACC[i] - pred_wacc) / act_WACC[i]))
    abs_errors.append(abs(act_WACC[i] - pred_wacc))

# Compute metrics
r_value_final = calculate_r_value(B_WACC_best, act_WACC)
mape = np.mean(abs_perc_errors) * 100  # in percentage
aae = np.mean(abs_errors)

# Print metrics
print("\nEvaluation for Best Technology Premium Combination (Highest R-value):")
print("R-value:", round(r_value_final, 4))
print("MAPE (%):", round(mape, 2))
print("AAE:", round(aae, 4))
