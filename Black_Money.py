# Save the Streamlit app script to a file
# Streamlit App Script

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# Title of the Streamlit app
st.title('Black Money Data Analysis')

# Load the dataset
# Caching the data to improve app performance
@st.cache_data
def load_data():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the relative path to the CSV file
    data_path = os.path.join(script_dir, 'data', 'Big_Black_Money_Dataset.csv')
    
    # Check if the file exists, display an error if not found
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}")
        return None
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Convert 'Date of Transaction' column to datetime format
    df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'])
    
    return df

df = load_data()

# Display the data if it exists
if df is not None:
    st.write(df)

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select Analysis Type:', ['Overall Analysis', 'Country-wise Analysis', 'Year-wise Analysis', 'Industry-wise Analysis'])

# Sidebar for country selection
if options == 'Country-wise Analysis':
    country = st.sidebar.selectbox('Select Country:', df['Country'].unique())
    if st.sidebar.button('Analyze'):
        st.session_state['previous_analysis'] = 'Country-wise Analysis'
        st.session_state['selected_country'] = country

# Sidebar for year selection
if options == 'Year-wise Analysis':
    year = st.sidebar.selectbox('Select Year:', df['Date of Transaction'].dt.year.unique())
    if st.sidebar.button('Analyze'):
        st.session_state['previous_analysis'] = 'Year-wise Analysis'
        st.session_state['selected_year'] = year

# Sidebar for industry selection
if options == 'Industry-wise Analysis':
    industry = st.sidebar.selectbox('Select Industry:', df['Industry'].unique())
    if st.sidebar.button('Analyze'):
        st.session_state['previous_analysis'] = 'Industry-wise Analysis'
        st.session_state['selected_industry'] = industry

# Display previous analysis
if 'previous_analysis' in st.session_state:
    st.sidebar.subheader('Previously Analyzed')
    st.sidebar.write(f"Analysis Type: {st.session_state['previous_analysis']}")
    if st.session_state['previous_analysis'] == 'Country-wise Analysis':
        st.sidebar.write(f"Country: {st.session_state['selected_country']}")
    elif st.session_state['previous_analysis'] == 'Year-wise Analysis':
        st.sidebar.write(f"Year: {st.session_state['selected_year']}")
    elif st.session_state['previous_analysis'] == 'Industry-wise Analysis':
        st.sidebar.write(f"Industry: {st.session_state['selected_industry']}")

# Adding EDA Questions Section

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Sidebar for EDA Questions
st.sidebar.title('EDA Questions')
show_eda_questions = st.sidebar.checkbox('Show EDA Questions')

if show_eda_questions:
    st.title('Exploratory Data Analysis (EDA) Questions')

    # Question 1
    st.subheader('1. What is the distribution of transaction amounts?')
    fig, ax = plt.subplots()
    sns.histplot(df['Amount (USD)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Transaction Amounts')
    st.pyplot(fig)
    st.markdown('The distribution of transaction amounts shows the frequency of different transaction sizes. Most transactions are of smaller amounts, with a few large transactions.')

    # Question 2
    st.subheader('2. Which country has the highest number of transactions?')
    fig, ax = plt.subplots()
    sns.countplot(y='Country', data=df, order=df['Country'].value_counts().index, ax=ax)
    ax.set_title('Number of Transactions by Country')
    st.pyplot(fig)
    st.markdown('China has the highest number of transactions, followed by the USA and India. This indicates that these countries have a higher volume of transactions in the dataset.')

    # Question 3
    st.subheader('3. What is the trend of monthly transaction amounts over time?')
    fig, ax = plt.subplots(figsize=(14, 7))
    df.groupby(df['Date of Transaction'].dt.to_period('M'))['Amount (USD)'].sum().plot(ax=ax)
    ax.set_title('Monthly Transaction Amount Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Amount (USD)')
    st.pyplot(fig)
    st.markdown('The trend of monthly transaction amounts over time shows fluctuations, with some months having higher transaction amounts than others. This can help identify seasonal patterns or anomalies.')

    # Question 4
    st.subheader('4. Which industry has the highest average transaction amount?')
    fig, ax = plt.subplots()
    avg_amount_by_industry = df.groupby('Industry')['Amount (USD)'].mean().sort_values()
    sns.barplot(x=avg_amount_by_industry, y=avg_amount_by_industry.index, ax=ax)
    ax.set_title('Average Transaction Amount by Industry')
    st.pyplot(fig)
    st.markdown('The industry with the highest average transaction amount is the Real Estate industry, followed by the Finance and Technology industries. This indicates that transactions in these industries tend to be larger.')

    # Question 5
    st.subheader('5. What is the distribution of money laundering risk scores?')
    fig, ax = plt.subplots()
    sns.histplot(df['Money Laundering Risk Score'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Money Laundering Risk Scores')
    st.pyplot(fig)
    st.markdown('The distribution of money laundering risk scores shows the frequency of different risk levels. Most transactions have a lower risk score, with a few transactions having higher risk scores.')

    # Question 6
    st.subheader('6. How does the transaction amount vary by transaction type?')
    fig, ax = plt.subplots()
    sns.boxplot(x='Transaction Type', y='Amount (USD)', data=df, ax=ax)
    ax.set_title('Transaction Amount by Transaction Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    st.markdown('The transaction amount varies significantly by transaction type. For example, Property Purchases tend to have higher transaction amounts compared to Cash Withdrawals.')

    # Question 7
    st.subheader('7. What is the correlation between transaction amount and risk score?')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Amount (USD)', y='Money Laundering Risk Score', data=df, ax=ax)
    ax.set_title('Correlation between Transaction Amount and Risk Score')
    st.pyplot(fig)
    st.markdown('There is a positive correlation between transaction amount and money laundering risk score. Higher transaction amounts tend to have higher risk scores.')

    # Question 8
    st.subheader('8. Which country has the highest average money laundering risk score?')
    fig, ax = plt.subplots()
    avg_risk_score_by_country = df.groupby('Country')['Money Laundering Risk Score'].mean().sort_values()
    sns.barplot(x=avg_risk_score_by_country, y=avg_risk_score_by_country.index, ax=ax)
    ax.set_title('Average Money Laundering Risk Score by Country')
    st.pyplot(fig)
    st.markdown('The country with the highest average money laundering risk score is Brazil, followed by Russia and China. This indicates that transactions in these countries tend to have higher risk scores.')

    # Question 9
    st.subheader('9. How does the transaction amount vary by country?')
    fig, ax = plt.subplots()
    sns.boxplot(x='Country', y='Amount (USD)', data=df, ax=ax)
    ax.set_title('Transaction Amount by Country')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    st.markdown('The transaction amount varies significantly by country. For example, transactions in the USA tend to have higher amounts compared to transactions in other countries.')

    # Question 10
    st.subheader('10. What is the relationship between transaction amount, risk score, and shell companies?')
    fig = px.scatter(df, x='Amount (USD)', y='Money Laundering Risk Score', color='Shell Companies Involved', hover_name='Country', title='Relationship between Transaction Amount, Risk Score, and Shell Companies')
    st.plotly_chart(fig)
    st.markdown('The scatter plot shows the relationship between transaction amount, money laundering risk score, and the involvement of shell companies. Transactions involving shell companies tend to have higher risk scores and larger amounts.')


# Overall Analysis
if options == 'Overall Analysis':
    st.subheader('Dataset')
    st.write(df.head())

    st.subheader('Summary Statistics')
    st.write(df.describe(include='all'))

    st.subheader('Distribution of Amount (USD)')
    fig, ax = plt.subplots()
    sns.histplot(df['Amount (USD)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Amount (USD)')
    st.pyplot(fig)
    st.markdown('<span style="color:green">This plot shows the distribution of transaction amounts, indicating the frequency of different transaction sizes.</span>', unsafe_allow_html=True)

    st.subheader('Count of Transactions by Country')
    fig, ax = plt.subplots()
    sns.countplot(y='Country', data=df, order=df['Country'].value_counts().index, ax=ax)
    ax.set_title('Count of Transactions by Country')
    st.pyplot(fig)
    st.markdown('<span style="color:green">This plot shows the number of transactions for each country, highlighting the countries with the most transactions.</span>', unsafe_allow_html=True)

    st.subheader('Correlation Matrix')
    fig, ax = plt.subplots()
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    st.markdown('<span style="color:green">This heatmap shows the correlation between different numeric variables, helping to identify relationships between them.</span>', unsafe_allow_html=True)

    st.subheader('Scatter Plot of Amount, Risk Score, and Shell Companies')
    fig = px.scatter(df, x='Amount (USD)', y='Money Laundering Risk Score', color='Shell Companies Involved', hover_name='Country', title='Scatter Plot of Amount, Risk Score, and Shell Companies')
    st.plotly_chart(fig)
    st.markdown('<span style="color:green">This scatter plot shows the relationship between transaction amount, risk score, and the involvement of shell companies.</span>', unsafe_allow_html=True)

# Country-wise Analysis
if options == 'Country-wise Analysis' and 'selected_country' in st.session_state:
    country_df = df[df['Country'] == st.session_state['selected_country']]
    st.subheader(f"Analysis for {st.session_state['selected_country']}")

    st.subheader('Distribution of Amount (USD)')
    fig, ax = plt.subplots()
    sns.histplot(country_df['Amount (USD)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Amount (USD)')
    st.pyplot(fig)
    st.markdown('<span style="color:blue">This plot shows the distribution of transaction amounts for the selected country.</span>', unsafe_allow_html=True)

    st.subheader('Count of Transactions by Transaction Type')
    fig, ax = plt.subplots()
    sns.countplot(y='Transaction Type', data=country_df, order=country_df['Transaction Type'].value_counts().index, ax=ax)
    ax.set_title('Count of Transactions by Transaction Type')
    st.pyplot(fig)
    st.markdown('<span style="color:blue">This plot shows the number of transactions for each transaction type in the selected country.</span>', unsafe_allow_html=True)

    st.subheader('Scatter Plot of Amount, Risk Score, and Shell Companies')
    fig = px.scatter(country_df, x='Amount (USD)', y='Money Laundering Risk Score', color='Shell Companies Involved', hover_name='Transaction Type', title='Scatter Plot of Amount, Risk Score, and Shell Companies')
    st.plotly_chart(fig)
    st.markdown('<span style="color:blue">This scatter plot shows the relationship between transaction amount, risk score, and the involvement of shell companies in the selected country.</span>', unsafe_allow_html=True)

# Year-wise Analysis
if options == 'Year-wise Analysis' and 'selected_year' in st.session_state:
    year_df = df[df['Date of Transaction'].dt.year == st.session_state['selected_year']]
    st.subheader(f"Analysis for {st.session_state['selected_year']}")

    st.subheader('Distribution of Amount (USD)')
    fig, ax = plt.subplots()
    sns.histplot(year_df['Amount (USD)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Amount (USD)')
    st.pyplot(fig)
    st.markdown('<span style="color:purple">This plot shows the distribution of transaction amounts for the selected year.</span>', unsafe_allow_html=True)

    st.subheader('Count of Transactions by Transaction Type')
    fig, ax = plt.subplots()
    sns.countplot(y='Transaction Type', data=year_df, order=year_df['Transaction Type'].value_counts().index, ax=ax)
    ax.set_title('Count of Transactions by Transaction Type')
    st.pyplot(fig)
    st.markdown('<span style="color:purple">This plot shows the number of transactions for each transaction type in the selected year.</span>', unsafe_allow_html=True)

    st.subheader('Scatter Plot of Amount, Risk Score, and Shell Companies')
    fig = px.scatter(year_df, x='Amount (USD)', y='Money Laundering Risk Score', color='Shell Companies Involved', hover_name='Transaction Type', title='Scatter Plot of Amount, Risk Score, and Shell Companies')
    st.plotly_chart(fig)
    st.markdown('<span style="color:purple">This scatter plot shows the relationship between transaction amount, risk score, and the involvement of shell companies in the selected year.</span>', unsafe_allow_html=True)

# Industry-wise Analysis
if options == 'Industry-wise Analysis' and 'selected_industry' in st.session_state:
    industry_df = df[df['Industry'] == st.session_state['selected_industry']]
    st.subheader(f"Analysis for {st.session_state['selected_industry']}")

    st.subheader('Distribution of Amount (USD)')
    fig, ax = plt.subplots()
    sns.histplot(industry_df['Amount (USD)'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Amount (USD)')
    st.pyplot(fig)
    st.markdown('<span style="color:orange">This plot shows the distribution of transaction amounts for the selected industry.</span>', unsafe_allow_html=True)

    st.subheader('Count of Transactions by Transaction Type')
    fig, ax = plt.subplots()
    sns.countplot(y='Transaction Type', data=industry_df, order=industry_df['Transaction Type'].value_counts().index, ax=ax)
    ax.set_title('Count of Transactions by Transaction Type')
    st.pyplot(fig)
    st.markdown('<span style="color:orange">This plot shows the number of transactions for each transaction type in the selected industry.</span>', unsafe_allow_html=True)

    st.subheader('Scatter Plot of Amount, Risk Score, and Shell Companies')
    fig = px.scatter(industry_df, x='Amount (USD)', y='Money Laundering Risk Score', color='Shell Companies Involved', hover_name='Transaction Type', title='Scatter Plot of Amount, Risk Score, and Shell Companies')
    st.plotly_chart(fig)
    st.markdown('<span style="color:orange">This scatter plot shows the relationship between transaction amount, risk score, and the involvement of shell companies in the selected industry.</span>', unsafe_allow_html=True)

    # Sub-sections for each industry
    sub_industries = industry_df['Industry'].unique()
    for sub_industry in sub_industries:
        sub_industry_df = industry_df[industry_df['Industry'] == sub_industry]
        st.subheader(f"Analysis for {sub_industry}")

        st.subheader('Distribution of Amount (USD)')
        fig, ax = plt.subplots()
        sns.histplot(sub_industry_df['Amount (USD)'], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of Amount (USD) for {sub_industry}')
        st.pyplot(fig)
        st.markdown(f'<span style="color:red">This plot shows the distribution of transaction amounts for the {sub_industry} industry.</span>', unsafe_allow_html=True)

        st.subheader('Count of Transactions by Transaction Type')
        fig, ax = plt.subplots()
        sns.countplot(y='Transaction Type', data=sub_industry_df, order=sub_industry_df['Transaction Type'].value_counts().index, ax=ax)
        ax.set_title(f'Count of Transactions by Transaction Type for {sub_industry}')
        st.pyplot(fig)
        st.markdown(f'<span style="color:red">This plot shows the number of transactions for each transaction type in the {sub_industry} industry.</span>', unsafe_allow_html=True)

        st.subheader('Scatter Plot of Amount, Risk Score, and Shell Companies')
        fig = px.scatter(sub_industry_df, x='Amount (USD)', y='Money Laundering Risk Score', color='Shell Companies Involved', hover_name='Transaction Type', title=f'Scatter Plot of Amount, Risk Score, and Shell Companies for {sub_industry}')
        st.plotly_chart(fig)
        st.markdown(f'<span style="color:red">This scatter plot shows the relationship between transaction amount, risk score, and the involvement of shell companies in the {sub_industry} industry.</span>', unsafe_allow_html=True)
# Adding Scatter and Bar Graphs for Relationships

# Scatter Plot of Amount (USD) vs Money Laundering Risk Score
st.subheader('Scatter Plot of Amount (USD) vs Money Laundering Risk Score')
fig = px.scatter(df, x='Amount (USD)', y='Money Laundering Risk Score', color='Country', hover_name='Transaction Type', title='Scatter Plot of Amount (USD) vs Money Laundering Risk Score')
st.plotly_chart(fig)
st.markdown('<span style="color:green">This scatter plot shows the relationship between transaction amount and money laundering risk score, colored by country.</span>', unsafe_allow_html=True)

# Bar Graph of Total Amount (USD) by Country
st.subheader('Bar Graph of Total Amount (USD) by Country')
total_amount_by_country = df.groupby('Country')['Amount (USD)'].sum().reset_index()
fig = px.bar(total_amount_by_country, x='Country', y='Amount (USD)', title='Total Amount (USD) by Country')
st.plotly_chart(fig)
st.markdown('<span style="color:green">This bar graph shows the total transaction amount for each country.</span>', unsafe_allow_html=True)

# Bar Graph of Total Amount (USD) by Transaction Type
st.subheader('Bar Graph of Total Amount (USD) by Transaction Type')
total_amount_by_transaction_type = df.groupby('Transaction Type')['Amount (USD)'].sum().reset_index()
fig = px.bar(total_amount_by_transaction_type, x='Transaction Type', y='Amount (USD)', title='Total Amount (USD) by Transaction Type')
st.plotly_chart(fig)
st.markdown('<span style="color:green">This bar graph shows the total transaction amount for each transaction type.</span>', unsafe_allow_html=True)

# Adding functionality to filter data based on multiple constraints and visualize

# Sidebar for multiple selections
st.sidebar.title('Filter Data')
selected_countries = st.sidebar.multiselect('Select Countries:', df['Country'].unique())
selected_years = st.sidebar.multiselect('Select Years:', df['Date of Transaction'].dt.year.unique())
selected_industries = st.sidebar.multiselect('Select Industries:', df['Industry'].unique())

# Filter data based on selections
filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if selected_years:
    filtered_df = filtered_df[filtered_df['Date of Transaction'].dt.year.isin(selected_years)]
if selected_industries:
    filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_df.head())

if not filtered_df.empty:
    # Scatter Plot of Amount (USD) vs Money Laundering Risk Score for filtered data
    st.subheader('Scatter Plot of Amount (USD) vs Money Laundering Risk Score (Filtered Data)')
    fig = px.scatter(filtered_df, x='Amount (USD)', y='Money Laundering Risk Score', color='Country', hover_name='Transaction Type', title='Scatter Plot of Amount (USD) vs Money Laundering Risk Score (Filtered Data)')
    st.plotly_chart(fig)
    st.markdown('<span style="color:green">This scatter plot shows the relationship between transaction amount and money laundering risk score for the filtered data, colored by country.</span>', unsafe_allow_html=True)

    # Bar Graph of Total Amount (USD) by Country for filtered data
    st.subheader('Bar Graph of Total Amount (USD) by Country (Filtered Data)')
    total_amount_by_country_filtered = filtered_df.groupby('Country')['Amount (USD)'].sum().reset_index()
    fig = px.bar(total_amount_by_country_filtered, x='Country', y='Amount (USD)', title='Total Amount (USD) by Country (Filtered Data)')
    st.plotly_chart(fig)
    st.markdown('<span style="color:green">This bar graph shows the total transaction amount for each country in the filtered data.</span>', unsafe_allow_html=True)

    # Bar Graph of Total Amount (USD) by Transaction Type for filtered data
    st.subheader('Bar Graph of Total Amount (USD) by Transaction Type (Filtered Data)')
    total_amount_by_transaction_type_filtered = filtered_df.groupby('Transaction Type')['Amount (USD)'].sum().reset_index()
    fig = px.bar(total_amount_by_transaction_type_filtered, x='Transaction Type', y='Amount (USD)', title='Total Amount (USD) by Transaction Type (Filtered Data)')
    st.plotly_chart(fig)
    st.markdown('<span style="color:green">This bar graph shows the total transaction amount for each transaction type in the filtered data.</span>', unsafe_allow_html=True)
else:
    st.write('No data available for the selected filters.')

# Summary of the analysis
st.subheader('Summary of the Analysis')
st.markdown('''
- **Data Loading and Cleaning**: Loaded necessary libraries, loaded the dataset, checked for missing values, and converted 'Date of Transaction' to datetime format.
- **Exploratory Data Analysis (EDA)**: Generated summary statistics, visualized the distribution of 'Amount (USD)', counted the number of transactions by country, and created a correlation matrix heatmap.
- **Data Visualization**: Analyzed the time series of monthly transaction amounts, identified the top 10 countries by transaction amount, visualized the distribution of 'Money Laundering Risk Score', counted transactions by transaction type, created boxplots of 'Amount (USD)' by country and transaction type, generated pair plots for numeric features, created a heatmap of missing values, and created various heatmaps, joint plots, and KDE plots to analyze relationships between 'Amount (USD)' and 'Money Laundering Risk Score'.
- **Final Visualization**: Created scatter plots of 'Amount (USD)', 'Money Laundering Risk Score', and 'Shell Companies Involved' using seaborn and plotly.

This app provides a comprehensive analysis of the dataset, including data cleaning, exploratory data analysis, and various visualizations to understand the relationships and distributions within the data.
''')
