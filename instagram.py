import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats

# Page Config
st.set_page_config(page_title="Marketing ROI Analytics", layout="wide")
st.title("📊 Marketing Campaign ROI & Conversion Analytics")

# =========================
# 1. DATA GENERATION (Numpy/Pandas)
# =========================
@st.cache_data
def generate_dataset(n=1000):
    np.random.seed(42)
    channels = ['Facebook', 'Google', 'Instagram', 'Email', 'YouTube']
    regions = ['North', 'South', 'East', 'West']

    data = pd.DataFrame({
        'campaign_id': range(1, n+1),
        'channel': np.random.choice(channels, n),
        'region': np.random.choice(regions, n),
        'spend': np.random.uniform(1000, 10000, n),
        'clicks': np.random.randint(100, 5000, n),
        'leads': np.random.randint(50, 1000, n),
        'promo_code': np.random.choice(['PROMO10', 'SAVE20', 'NONE'], n)
    })
    
    # Logic: Conversions are a subset of leads, Revenue is a multiple of conversions
    data['conversions'] = (data['leads'] * np.random.uniform(0.1, 0.4, n)).astype(int)
    data['revenue'] = data['conversions'] * np.random.uniform(20, 150, n)
    return data

# File Upload Logic
uploaded_file = st.file_uploader("Upload Marketing Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_dataset()
    st.info("Using Auto-Generated Sample Dataset")

# =========================
# 2. DATA CLEANING & FEATURE ENGINEERING
# =========================
st.header("1️⃣ Data Processing")

# Cleaning
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Feature Engineering
df['ROI'] = (df['revenue'] - df['spend']) / df['spend']
df['CAC'] = df['spend'] / df['conversions'].replace(0, 1)
df['Conversion_Rate'] = df['conversions'] / df['clicks']
df['Promo_Used'] = df['promo_code'].apply(lambda x: 0 if x == "NONE" else 1)

# Regex for Promo Validation
df['Valid_Promo'] = df['promo_code'].apply(
    lambda x: 1 if re.match(r'^[A-Z]+\d+$', str(x)) else 0
)

# Manual Normalization (Standard Score: (x - mean) / std)
num_cols = ['spend', 'clicks', 'leads', 'conversions', 'revenue']
df_scaled = df.copy()
for col in num_cols:
    df_scaled[col] = (df[col] - df[col].mean()) / df[col].std()

st.dataframe(df.head())

# =========================
# 3. VISUAL ANALYSIS (Matplotlib)
# =========================
st.header("2️⃣ Performance Visuals")
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROI by Channel")
    roi_data = df.groupby('channel')['ROI'].mean().sort_values()
    fig, ax = plt.subplots()
    ax.barh(roi_data.index, roi_data.values, color='#4CAF50')
    ax.set_xlabel('Average ROI')
    st.pyplot(fig)

with col2:
    st.subheader("Acquisition Cost (CAC)")
    cac_data = df.groupby('channel')['CAC'].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    ax2.bar(cac_data.index, cac_data.values, color='#FF5722')
    ax2.set_ylabel('Avg Cost Per Conversion')
    st.pyplot(fig2)

# =========================
# 4. STATISTICAL INFERENCE (Scipy)
# =========================
st.header("3️⃣ Hypothesis Testing")
promo_conv = df[df['Promo_Used'] == 1]['Conversion_Rate']
none_conv = df[df['Promo_Used'] == 0]['Conversion_Rate']

t_stat, p_val = stats.ttest_ind(promo_conv, none_conv)

st.write(f"**T-Statistic:** {t_stat:.4f} | **P-Value:** {p_val:.4f}")
if p_val < 0.05:
    st.success("Result: Promo codes significantly impact conversion rates.")
else:
    st.warning("Result: No statistically significant impact from promo codes.")

# =========================
# 5. PREDICTIVE MODELING (Numpy OLS)
# =========================
st.header("4️⃣ Revenue Prediction Model")
st.write("Calculated using the Normal Equation: $\\hat{\\beta} = (X^T X)^{-1} X^T y$")

# Prepare Data
X = df[['spend', 'clicks', 'leads', 'Promo_Used']].values
X = np.hstack([np.ones((X.shape[0], 1)), X]) # Add Intercept
y = df['revenue'].values

# Matrix Calculations
try:
    # beta = (X.T * X)^-1 * X.T * y
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    
    # Metrics
    preds = X @ beta
    residuals = y - preds
    r2 = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
    
    # Display Results
    cols = st.columns(5)
    labels = ['Intercept', 'Spend', 'Clicks', 'Leads', 'Promo']
    for i, col in enumerate(cols):
        col.metric(labels[i], f"{beta[i]:.2f}")
    
    st.write(f"**Model R-Squared Score:** `{r2:.4f}`")
except np.linalg.LinAlgError:
    st.error("Error: Matrix is singular. Check for multicollinearity in your data.")

st.balloons()
