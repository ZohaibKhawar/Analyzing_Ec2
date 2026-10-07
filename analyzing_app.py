import io
from threading import RLock

import streamlit as st
import pandas as pd
from matplotlib.artist import setp
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("EC2 Instance Cost Analysis")

# Load the dataset
file_path = 'ec2dataset.csv'
data = pd.read_csv(file_path)

# Set up the plotting style
sns.set(style="whitegrid")


# Matplotlib is not thread-safe, so only one browser tab may draw a chart at a time.
# The charts also use Figure() instead of plt.figure(), because Streamlit closes
# every pyplot figure whenever any tab finishes a run.
@st.cache_resource
def get_plot_lock():
    return RLock()


plot_lock = get_plot_lock()

# ---------------------------------------------------------
# Part 1: Cost Analysis
# ---------------------------------------------------------
st.header("Part 1: Cost Analysis")

st.subheader("Dataset Information")

# data.info() prints to the terminal, so capture its text for the page
buffer = io.StringIO()
data.info(buf=buffer)
st.code(buffer.getvalue(), language=None)

st.subheader("Dataset Preview")
st.dataframe(data.head())

# List of cost-related columns
cost_columns = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost', 'Windows On Demand cost', 'Windows Reserved cost']

# Remove '$' and 'hourly' and convert columns to numeric
for column in cost_columns:
    data[column] = pd.to_numeric(data[column].str.replace('[$, hourly]', '', regex=True), errors='coerce')

# Check for missing values after conversion
st.subheader("Missing Values After Cleaning")
st.dataframe(data[cost_columns].isnull().sum().rename("Missing Values"))

# Generate summary statistics for cost-related columns
st.subheader("Cost Summary Statistics")
cost_summary = data[cost_columns].describe()
st.dataframe(cost_summary)

# Create a boxplot to visualize the distribution of costs
st.subheader("Cost Distribution")
with plot_lock:
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    sns.boxplot(data=data[cost_columns], palette="Set2", ax=ax)

    # Set the plot labels and title
    ax.set_title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
    ax.set_ylabel('Cost (USD)', fontsize=12)
    setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    # Use a log scale so the outliers do not squash the boxes
    ax.set_yscale('log')

    # Display the plot
    fig.tight_layout()
    st.pyplot(fig)


# Function to identify outliers using the IQR method
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]


# Find outliers in the On-Demand cost column
st.subheader("On-Demand Cost Outliers")
outliers_on_demand = detect_outliers('On Demand')
st.write(f"{len(outliers_on_demand)} outliers found")
st.dataframe(outliers_on_demand)

# Compare Reserved and On-Demand costs
st.subheader("Lowest-Cost Instances: On-Demand vs Reserved")
cost_comparison = data[['Name', 'On Demand', 'Linux Reserved cost']].dropna().sort_values('On Demand')
st.dataframe(cost_comparison.head(10))  # Display the top 10 cost-effective instances

# ---------------------------------------------------------
# Instance Families
# ---------------------------------------------------------
st.header("Instance Families: T2 vs T3")


# Filter for specific instance families
def filter_instance_family(family):
    return data[data['Name'].str.startswith(family)]


# Example: Filter for T2 and T3 instance families
t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')

# Compare summary statistics for T2 and T3 instances
t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

st.subheader("T2 Instance Costs Summary")
st.dataframe(t2_summary)

st.subheader("T3 Instance Costs Summary")
st.dataframe(t3_summary)

# Visualize cost comparison for T2 and T3 instance families
# Add T2 instances
st.subheader("Cost Distribution for T2 Instances")
with plot_lock:
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    sns.boxplot(data=t2_instances[cost_columns], palette="Blues", showmeans=True, ax=ax)
    ax.set_title('Cost Distribution for T2 Instances', fontsize=16)
    ax.set_ylabel('Cost (USD)', fontsize=12)
    setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)

# Add T3 instances
st.subheader("Cost Distribution for T3 Instances")
with plot_lock:
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    sns.boxplot(data=t3_instances[cost_columns], palette="Greens", showmeans=True, ax=ax)
    ax.set_title('Cost Distribution for T3 Instances', fontsize=16)
    ax.set_ylabel('Cost (USD)', fontsize=12)
    setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)

# Compare On-Demand and Reserved costs for T2 and T3 families
st.subheader("Lowest-Cost T2 and T3 Instances")
comparison = pd.concat([t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]])
# Sort by On-Demand costs
comparison_sorted = comparison.dropna().sort_values('On Demand')
st.dataframe(comparison_sorted.head(10))  # Display the 10 lowest-cost instances

# ---------------------------------------------------------
# Part 2: Predicting On-Demand Cost
# ---------------------------------------------------------
st.header("Part 2: Predicting On-Demand Cost")

# Convert memory from string (e.g., '0.5 GiB') to numeric by stripping ' GiB'
data['Instance Memory'] = pd.to_numeric(data['Instance Memory'].str.replace(' GiB', ''))
# Extract the number of vCPUs from the 'vCPUs' column (e.g., '2 vCPUs')
data['vCPUs'] = pd.to_numeric(data['vCPUs'].str.extract(r'(\d+)', expand=False))

# Check the data after conversion
st.subheader("Features After Conversion")
st.dataframe(data[['Instance Memory', 'vCPUs']].head())

# Drop rows with missing values in the relevant columns (for simplicity)
data_cleaned = data.dropna(subset=['On Demand', 'Instance Memory', 'vCPUs'])

# Verify that the cleaned data has no missing values
st.subheader("Missing Values in Cleaned Data")
st.dataframe(data_cleaned.isnull().sum().rename("Missing Values"))

# Remove the On-Demand cost outliers found in Part 1, they pull the predictions below $0
st.subheader("Outliers Removed")
data_cleaned = data_cleaned[~data_cleaned.index.isin(outliers_on_demand.index)]
st.write(f"{len(outliers_on_demand)} outliers removed, {len(data_cleaned)} instances left for the model")

# Define features and target
X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Train/Test Split")

col1, col2 = st.columns(2)

col1.metric("Training Samples", len(X_train))
col2.metric("Testing Samples", len(X_test))

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Linear Regression Model")

col1, col2, col3 = st.columns(3)

col1.metric("Intercept", f"{model.intercept_:.4f}")
col2.metric("Memory Coefficient", f"{model.coef_[0]:.4f}")
col3.metric("vCPU Coefficient", f"{model.coef_[1]:.4f}")

# Predict the costs for the test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
col2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

# Plot actual vs predicted values
st.subheader("Actual vs Predicted On-Demand Costs")
with plot_lock:
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7, color='b')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_title('Actual vs Predicted On-Demand Costs')
    ax.set_xlabel('Actual On-Demand Cost')
    ax.set_ylabel('Predicted On-Demand Cost')
    st.pyplot(fig)

# Predict cost for a new instance (defaults to the example: 4 GiB memory and 2 vCPUs)
st.subheader("Predict the Cost of a New Instance")

col1, col2 = st.columns(2)

memory = col1.number_input("Memory (GiB)", min_value=0.5, value=4.0, step=0.5)
vcpus = col2.number_input("vCPUs", min_value=1, value=2, step=1)

new_instance = pd.DataFrame([[memory, vcpus]], columns=['Instance Memory', 'vCPUs'])
predicted_cost = model.predict(new_instance)

st.metric(
    f"Predicted On-Demand Cost for {memory:g} GiB, {vcpus} vCPUs",
    f"${predicted_cost[0]:.4f} hourly"
)
