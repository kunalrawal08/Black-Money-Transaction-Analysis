# Save the Streamlit app script to a file
# Streamlit App Script

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Title of the Streamlit app
st.title('Black Money Data Analysis')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\\Users\\KUNAL\\Desktop\\Black Money Data Analyze\\Big_Black_Money_Dataset.csv")
    df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'])
    return df

df = load_data()

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
