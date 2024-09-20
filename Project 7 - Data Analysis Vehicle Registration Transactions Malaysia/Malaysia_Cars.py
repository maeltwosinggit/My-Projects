import pandas as pd
import plotly.express as px
import streamlit as st

# Initialize an empty list to store DataFrames
dfs = []

# Loop through the years 2000 to 2024
for year in range(2000, 2025):
    # Generate the URL for each parquet file
    URL_DATA = f'https://storage.data.gov.my/transportation/cars_{year}.parquet'
    
    # Read the parquet file into a DataFrame
    df = pd.read_parquet(URL_DATA)
    
    # Convert 'date' to datetime if the column exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Convert 'date_reg' column to datetime, assuming it's in nanoseconds
    df['date_reg'] = pd.to_datetime(df['date_reg'], unit='ns')
    
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Convert 'date_reg' to datetime
final_df['date_reg'] = pd.to_datetime(final_df['date_reg'])

# Group by 'date_reg' and 'type', then count occurrences
model_counts = final_df.groupby(['date_reg', 'type']).size().reset_index(name='counts')

# Sort by 'date_reg' to ensure correct cumulative calculation
model_counts = model_counts.sort_values('date_reg')

# Calculate cumulative counts for each type
model_counts['cumulative_counts'] = model_counts.groupby('type')['counts'].cumsum()

# Calculate total counts per type for sorting
total_counts = model_counts.groupby('type')['cumulative_counts'].max().sort_values(ascending=False)
sorted_makers = total_counts.index

# Create a line plot using Plotly Express
fig = px.line(
    model_counts,
    x='date_reg',
    y='cumulative_counts',
    color='type',
    title='Cumulative Car Type Popularity Over Time',
    labels={'date_reg': 'Registration Date', 'cumulative_counts': 'Cumulative Number of Cars Registered'},
    category_orders={'type': sorted_makers}  # Sort legend based on total counts
)

# Set up Streamlit
st.title('Cumulative Car Type Popularity Over Time')

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)

# # Pivot the data to make it suitable for st.line_chart
# pivot_table = model_counts.pivot(index='date_reg', columns='type', values='cumulative_counts').fillna(0)

# # Use Streamlit's built-in line chart function
# st.subheader('Cumulative Car Type Popularity Using Streamlit\'s Line Chart')
# st.line_chart(model_counts,
#     x='date_reg',
#     y='cumulative_counts')
