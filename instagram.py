import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(page_title="Instagram Business Analytics", layout="wide")
st.title("📊 Instagram Performance Dashboard")
st.markdown("Analysis of Content Efficiency and Audience Engagement")

# --- 2. GENERATE SYNTHETIC DATA (Business Context) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=30)
    post_types = ['Reel', 'Carousel', 'Image']
    
    data = []
    for date in dates:
        p_type = np.random.choice(post_types)
        reach = np.random.randint(1000, 5000)
        impressions = reach * np.random.uniform(1.1, 1.5)
        likes = reach * np.random.uniform(0.02, 0.08)
        comments = likes * np.random.uniform(0.05, 0.1)
        saves = reach * np.random.uniform(0.01, 0.03)
        shares = reach * np.random.uniform(0.005, 0.02)
        visits = reach * np.random.uniform(0.02, 0.05)
        clicks = visits * np.random.uniform(0.1, 0.3)
        
        data.append([date, p_type, impressions, reach, likes, comments, saves, shares, visits, clicks])

    df = pd.DataFrame(data, columns=[
        'Date', 'Post_Type', 'Impressions', 'Reach', 'Likes', 
        'Comments', 'Saves', 'Shares', 'Profile_Visits', 'Website_Clicks'
    ])
    
    # Calculate Business KPIs
    df['Engagement_Score'] = (df['Likes'] * 1) + (df['Comments'] * 2) + (df['Saves'] * 3) + (df['Shares'] * 4)
    df['ER_Reach'] = (df['Likes'] + df['Comments'] + df['Saves'] + df['Shares']) / df['Reach'] * 100
    return df

df = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Filter Analytics")
selected_type = st.sidebar.multiselect("Select Post Type", options=df['Post_Type'].unique(), default=df['Post_Type'].unique())
date_range = st.sidebar.date_input("Date Range", [df['Date'].min(), df['Date'].max()])

# Apply Filters
mask = (df['Post_Type'].isin(selected_type)) & (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
filtered_df = df.loc[mask]

# --- 4. TOP LEVEL KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
total_reach = filtered_df['Reach'].sum()
avg_er = filtered_df['ER_Reach'].mean()
total_clicks = filtered_df['Website_Clicks'].sum()
conversion_rate = (total_clicks / filtered_df['Profile_Visits'].sum()) * 100

col1.metric("Total Reach", f"{total_reach/1000:.1f}K")
col2.metric("Avg. Engagement Rate", f"{avg_er:.2f}%")
col3.metric("Website Clicks", int(total_clicks))
col4.metric("Profile Conv. Rate", f"{conversion_rate:.1f}%")

st.markdown("---")

# --- 5. VISUALIZATIONS ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Reach vs. Impressions Over Time")
    fig_line = px.line(filtered_df, x='Date', y=['Reach', 'Impressions'], 
                       color_discrete_sequence=['#E1306C', '#FFDC80'], template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

with row1_col2:
    st.subheader("Engagement Quality by Post Type")
    # Grouping for bar chart
    avg_by_type = filtered_df.groupby('Post_Type')['ER_Reach'].mean().reset_index()
    fig_bar = px.bar(avg_by_type, x='Post_Type', y='ER_Reach', color='Post_Type',
                     labels={'ER_Reach': 'Avg Engagement Rate (%)'},
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Virality Correlation (Saves vs. Shares)")
    fig_scatter = px.scatter(filtered_df, x='Saves', y='Shares', size='Reach', color='Post_Type',
                             hover_name='Post_Type', log_x=True, template="ggplot2")
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_col2:
    st.subheader("Top Performing Content (Weighted)")
    top_posts = filtered_df.sort_values(by='Engagement_Score', ascending=False).head(5)
    st.dataframe(top_posts[['Date', 'Post_Type', 'Reach', 'Engagement_Score']], use_container_width=True)

# --- 6. STRATEGIC INSIGHTS ---
st.info(f"""
**Business Insight:** - The highest conversion rate is currently coming from **{avg_by_type.loc[avg_by_type['ER_Reach'].idxmax(), 'Post_Type']}** content. 
- A high correlation between Saves and Shares suggests your content is providing **Educational Value**.
""")
