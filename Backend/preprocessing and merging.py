import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# 1. DATA LOADING
print("1. DATA LOADING")
print("-" * 80)

def load_data():
    """
    Load all data files and return them as separate dataframes
    """
    # Check if files exist
    required_files = ['cleaned_balance_data.csv', 'cleaned_subregion_data.csv', 'cleaned_texas_data.csv']
    json_files = ['dallas.json', 'houston.json', 'la.json', 'nyc.json', 'philadelphia.json',
                 'phoenix.json', 'san_antonio.json', 'san_diego.json', 'san_jose.json', 'seattle.json']

    print("Loading CSV files...")
    # Load CSV files
    balance_df = pd.read_csv('cleaned_balance_data.csv')
    subregion_df = pd.read_csv('cleaned_subregion_data.csv')
    texas_df = pd.read_csv('cleaned_texas_data.csv')

    print("Loading JSON weather data...")
    # Load JSON weather data
    weather_data = {}
    for city_json in json_files:
        city_name = city_json.split('.')[0]  # Extract city name from filename
        try:
            with open(city_json, 'r') as f:
                city_data = json.load(f)
                # If it's a dictionary, convert to list
                if isinstance(city_data, dict):
                    city_data = [city_data]
                weather_data[city_name] = city_data
        except FileNotFoundError:
            print(f"Warning: {city_json} not found.")

    return balance_df, subregion_df, texas_df, weather_data

# Load data
balance_df, subregion_df, texas_df, weather_data = load_data()

# Display sample data
print("\nSample data from cleaned_balance_data.csv:")
print(balance_df.head())
print("\nSample data from cleaned_subregion_data.csv:")
print(subregion_df.head())
print("\nSample data from cleaned_texas_data.csv:")
print(texas_df.head())

# Display one example of weather data
print("\nSample weather data from dallas.json:")
if 'dallas' in weather_data and weather_data['dallas']:
    print(weather_data['dallas'][0])

# 2. DATA PROCESSING AND MERGING
print("\n2. DATA PROCESSING AND MERGING")
print("-" * 80)

def process_weather_data(weather_data):
    """
    Process weather data from JSON files into a pandas dataframe
    """
    print("Processing weather data...")
    all_weather_rows = []

    for city, data_list in weather_data.items():
        for item in data_list:
            # Convert Unix timestamp to datetime
            if 'time' in item:
                dt = datetime.fromtimestamp(item['time'])
                row = {
                    'city': city,
                    'local_time': dt,
                    'temperature': item.get('temperature'),
                    'humidity': item.get('humidity', np.nan),
                    'windSpeed': item.get('windSpeed', np.nan),
                    'pressure': item.get('pressure', np.nan),
                    'precipIntensity': item.get('precipIntensity', np.nan),
                    'precipProbability': item.get('precipProbability', np.nan),
                    'cloudCover': item.get('cloudCover', np.nan),
                    'uvIndex': item.get('uvIndex', np.nan),
                    'visibility': item.get('visibility', np.nan)
                }
                all_weather_rows.append(row)

    # Create DataFrame from weather data
    weather_df = pd.DataFrame(all_weather_rows)
    if not weather_df.empty:
        # Convert humidity from decimal to percentage
        if 'humidity' in weather_df.columns:
            weather_df['humidity'] = weather_df['humidity'] * 100

        # Convert datetime to string format to match other dataframes
        weather_df['local_time'] = weather_df['local_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return weather_df

def process_texas_data(texas_df):
    """
    Process Texas data to match the format of other dataframes
    """
    print("Processing Texas data...")
    # Melt the dataframe to get city-specific data
    texas_melted = pd.melt(
        texas_df,
        id_vars=['date'],
        value_vars=['houston', 'san antonio', 'dallas'],
        var_name='city',
        value_name='demand'
    )

    # Rename columns for consistency
    texas_melted.rename(columns={'date': 'local_time'}, inplace=True)

    # Add a company column for consistency
    texas_melted['company'] = 'ERCOT'

    return texas_melted

def prepare_data_for_merge(balance_df, subregion_df, texas_melted, weather_df):
    """
    Prepare all dataframes for merging
    """
    print("Preparing dataframes for merging...")

    # Ensure consistent datetime formats
    for df in [balance_df, subregion_df, texas_melted]:
        if 'local_time' in df.columns and df['local_time'].dtype == object:
            # Try to convert to datetime if it's not already
            try:
                df['local_time'] = pd.to_datetime(df['local_time'])
            except:
                print(f"Warning: Could not convert local_time to datetime in dataframe")

    if 'local_time' in weather_df.columns:
        weather_df['local_time'] = pd.to_datetime(weather_df['local_time'])

    # Normalize city names to lowercase
    for df in [balance_df, subregion_df, texas_melted, weather_df]:
        if 'city' in df.columns:
            df['city'] = df['city'].str.lower()

    return balance_df, subregion_df, texas_melted, weather_df

def merge_dataframes(balance_df, subregion_df, texas_melted, weather_df):
    """
    Merge all dataframes into a unified dataset
    """
    print("Merging dataframes...")

    # Merge balance and subregion dataframes
    merged_df = pd.concat([balance_df, subregion_df], ignore_index=True)

    # Merge with Texas data
    merged_df = pd.concat([merged_df, texas_melted], ignore_index=True)

    # Merge with weather data
    final_df = pd.merge(
        merged_df,
        weather_df,
        on=['city', 'local_time'],
        how='left'
    )

    return final_df

# Process and merge data
weather_df = process_weather_data(weather_data)
texas_melted = process_texas_data(texas_df)
balance_df, subregion_df, texas_melted, weather_df = prepare_data_for_merge(balance_df, subregion_df, texas_melted, weather_df)
merged_df = merge_dataframes(balance_df, subregion_df, texas_melted, weather_df)
merged_df = merged_df.sample(n=10000, random_state=42).reset_index(drop=True)

print("\nMerged dataset shape:", merged_df.shape)
print("\nMerged dataset sample:")
print(merged_df.head())

# 3. MISSING VALUES HANDLING
print("\n3. MISSING VALUES HANDLING")
print("-" * 80)

def handle_missing_values(df):
    """
    Identify and handle missing values in the dataset
    """
    print("Checking for missing values...")

    # Check missing values
    missing_values = df.isnull().sum()
    print("Missing values by column:")
    print(missing_values[missing_values > 0])

    # Calculate percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("\nPercentage of missing values by column:")
    print(missing_percentage[missing_percentage > 0])

    # Strategy for handling missing values:
    # 1. Drop columns with too many missing values (>50%)
    columns_to_drop = missing_percentage[missing_percentage > 50].index.tolist()
    if columns_to_drop:
        print(f"\nDropping columns with >50% missing values: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    # 2. For remaining numerical columns, impute with median
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            # Group by city and impute with median
            if 'city' in df.columns:
                city_medians = df.groupby('city')[col].transform('median')
                df[col] = df[col].fillna(city_medians)

                # If still missing, use overall median
                overall_median = df[col].median()
                df[col] = df[col].fillna(overall_median)
            else:
                # If no city column, use overall median
                df[col] = df[col].fillna(df[col].median())

    # 3. For categorical columns, impute with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values after imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    return df

# Handle missing values
merged_df = handle_missing_values(merged_df)

# 4. FEATURE ENGINEERING
print("\n4. FEATURE ENGINEERING")
print("-" * 80)

def engineer_features(df):
    """
    Extract time-based features and create new features
    """
    print("Engineering features...")

    # Ensure datetime format
    if 'local_time' in df.columns and df['local_time'].dtype != 'datetime64[ns]':
        df['local_time'] = pd.to_datetime(df['local_time'])

    # Extract time-based features
    df['hour'] = df['local_time'].dt.hour
    df['day_of_week'] = df['local_time'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_name'] = df['local_time'].dt.day_name()
    df['month'] = df['local_time'].dt.month
    df['year'] = df['local_time'].dt.year
    df['day'] = df['local_time'].dt.day

    # Create season feature
    df['season'] = df['month'].apply(lambda x:
                                    'Winter' if x in [12, 1, 2] else
                                    'Spring' if x in [3, 4, 5] else
                                    'Summer' if x in [6, 7, 8] else
                                    'Fall')

    # Create weekday/weekend feature
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Create time of day feature
    df['time_of_day'] = df['hour'].apply(lambda x:
                                        'Night' if x < 6 else
                                        'Morning' if x < 12 else
                                        'Afternoon' if x < 18 else
                                        'Evening')

    # Create derivative features if enough data
    if 'demand' in df.columns and 'temperature' in df.columns:
        # Calculate temperature-demand ratio
        df['temp_demand_ratio'] = df['demand'] / (df['temperature'] + 1)  # Add 1 to avoid division by zero

    # Calculate peak hour flag
    if 'hour' in df.columns and 'demand' in df.columns:
        peak_hours = [17, 18, 19, 20]  # 5pm-8pm
        df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in peak_hours else 0)

    return df

# Engineer features
merged_df = engineer_features(merged_df)

print("\nDataset with engineered features:")
print(merged_df.head())
print("\nEngineered features list:")
print(merged_df.columns.tolist())

# 5. NORMALIZE/SCALE FEATURES
print("\n5. NORMALIZE/SCALE FEATURES")
print("-" * 80)

def normalize_features(df):
    """
    Normalize or scale continuous variables
    """
    print("Normalizing features...")

    # Identify numerical columns to normalize
    # Exclude binary, categorical, and datetime features
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Exclude specific columns
    exclude_cols = ['hour', 'day_of_week', 'month', 'year', 'day', 'is_weekend', 'is_peak_hour']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    print(f"Normalizing these numerical columns: {numerical_cols}")

    # Create a copy of the DataFrame to store normalized values
    df_normalized = df.copy()

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the numerical columns
    if numerical_cols:
        # Create a subset of the DataFrame without NaN values
        df_subset = df[numerical_cols].dropna()

        # Fit the scaler
        scaler.fit(df_subset)

        # Transform the data
        for col in numerical_cols:
            col_normalized = f"{col}_normalized"
            df_normalized[col_normalized] = np.nan

            # Only normalize non-NaN values
            mask = ~df[col].isna()
            df_normalized.loc[mask, col_normalized] = scaler.fit_transform(df.loc[mask, [col]]).flatten()

    return df_normalized

# Normalize features
merged_df = normalize_features(merged_df)

print("\nDataset with normalized features:")
print(merged_df.head())

# 6. AGGREGATION
print("\n6. AGGREGATION")
print("-" * 80)

def aggregate_data(df):
    """
    Compute daily and weekly summary statistics
    """
    print("Aggregating data...")

    # Check if we have required columns
    required_cols = ['local_time', 'demand', 'temperature']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing required columns for aggregation.")
        return df, pd.DataFrame(), pd.DataFrame()

    # Ensure datetime format
    if df['local_time'].dtype != 'datetime64[ns]':
        df['local_time'] = pd.to_datetime(df['local_time'])

    # Create date column for aggregation
    df['date'] = df['local_time'].dt.date

    # Daily aggregation
    print("Computing daily aggregations...")
    daily_agg = df.groupby(['city', 'date']).agg({
        'demand': ['mean', 'min', 'max', 'sum', 'std'],
        'temperature': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max'],
        'windSpeed': ['mean', 'max']
    }).reset_index()

    # Flatten multi-level column names
    daily_agg.columns = ['_'.join(col).strip('_') for col in daily_agg.columns.values]

    # Weekly aggregation
    print("Computing weekly aggregations...")
    df['week'] = df['local_time'].dt.isocalendar().week
    df['year'] = df['local_time'].dt.year

    weekly_agg = df.groupby(['city', 'year', 'week']).agg({
        'demand': ['mean', 'min', 'max', 'sum', 'std'],
        'temperature': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean'],
        'windSpeed': ['mean']
    }).reset_index()

    # Flatten multi-level column names
    weekly_agg.columns = ['_'.join(col).strip('_') for col in weekly_agg.columns.values]

    return df, daily_agg, weekly_agg

# Aggregate data
merged_df, daily_agg, weekly_agg = aggregate_data(merged_df)

print("\nDaily aggregation sample:")
print(daily_agg.head())
print("\nWeekly aggregation sample:")
print(weekly_agg.head())

# 7. ANOMALY & ERROR DETECTION
print("\n7. ANOMALY & ERROR DETECTION")
print("-" * 80)

def detect_anomalies(df, daily_agg):
    """
    Detect anomalies in the dataset using statistical methods and Isolation Forest
    """
    print("Detecting anomalies...")

    # 1. Check for physical impossibilities
    print("\nChecking for physically impossible values...")

    anomalies = pd.DataFrame()

    # Temperature anomalies (extreme values)
    if 'temperature' in df.columns:
        temp_anomalies = df[(df['temperature'] < -50) | (df['temperature'] > 130)]
        if not temp_anomalies.empty:
            print(f"Found {len(temp_anomalies)} temperature anomalies (< -50°F or > 130°F)")
            anomalies = pd.concat([anomalies, temp_anomalies])

    # Humidity anomalies (outside 0-100%)
    if 'humidity' in df.columns:
        humidity_anomalies = df[(df['humidity'] < 0) | (df['humidity'] > 100)]
        if not humidity_anomalies.empty:
            print(f"Found {len(humidity_anomalies)} humidity anomalies (< 0% or > 100%)")
            anomalies = pd.concat([anomalies, humidity_anomalies])

    # Wind speed anomalies (negative values or extremely high)
    if 'windSpeed' in df.columns:
        wind_anomalies = df[(df['windSpeed'] < 0) | (df['windSpeed'] > 150)]
        if not wind_anomalies.empty:
            print(f"Found {len(wind_anomalies)} wind speed anomalies (< 0 mph or > 150 mph)")
            anomalies = pd.concat([anomalies, wind_anomalies])

    # Demand anomalies (negative values)
    if 'demand' in df.columns:
        demand_anomalies = df[df['demand'] < 0]
        if not demand_anomalies.empty:
            print(f"Found {len(demand_anomalies)} demand anomalies (negative values)")
            anomalies = pd.concat([anomalies, demand_anomalies])

    # 2. Statistical anomaly detection (Z-score method)
    print("\nDetecting statistical anomalies using Z-score method...")

    # Function to detect anomalies based on z-score
    def detect_zscore_anomalies(df, column, threshold=3):
        if column in df.columns:
            # Group by city to account for different distributions
            if 'city' in df.columns:
                # Calculate z-scores by city
                df[f'{column}_zscore'] = df.groupby('city')[column].transform(
                    lambda x: np.abs((x - x.mean()) / x.std())
                )
            else:
                # Calculate overall z-scores
                df[f'{column}_zscore'] = np.abs((df[column] - df[column].mean()) / df[column].std())

            # Identify anomalies
            anomalies = df[df[f'{column}_zscore'] > threshold]
            return anomalies
        return pd.DataFrame()

    # Detect anomalies in key variables
    for column in ['demand', 'temperature', 'humidity', 'windSpeed']:
        if column in df.columns:
            col_anomalies = detect_zscore_anomalies(df, column)
            if not col_anomalies.empty:
                print(f"Found {len(col_anomalies)} anomalies in {column} using Z-score method")
                anomalies = pd.concat([anomalies, col_anomalies])

    # 3. Machine learning-based anomaly detection (Isolation Forest)
    print("\nDetecting anomalies using Isolation Forest...")

    # Select columns for anomaly detection
    ml_cols = [col for col in ['demand', 'temperature', 'humidity', 'windSpeed']
              if col in df.columns]

    if ml_cols:
        # Create a copy of the dataframe with only the selected columns
        df_ml = df[ml_cols].copy()

        # Drop rows with missing values
        df_ml = df_ml.dropna()

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        df_ml['anomaly'] = iso_forest.fit_predict(df_ml)

        # -1 represents anomalies
        ml_anomalies = df_ml[df_ml['anomaly'] == -1]

        if not ml_anomalies.empty:
            print(f"Found {len(ml_anomalies)} anomalies using Isolation Forest")
            # Get the original rows with anomalies
            ml_anomaly_indices = ml_anomalies.index
            anomalies = pd.concat([anomalies, df.loc[ml_anomaly_indices]])

    # 4. Demand spike detection
    print("\nDetecting demand spikes...")

    if 'demand' in df.columns and 'city' in df.columns and 'local_time' in df.columns:
        # Group by city and date
        df['date'] = df['local_time'].dt.date

        # Calculate daily percentage changes
        demand_changes = df.groupby(['city', 'date'])['demand'].mean().pct_change().reset_index()
        demand_changes.columns = ['city', 'date', 'demand_pct_change']

        # Identify days with extreme changes
        spike_days = demand_changes[
            (demand_changes['demand_pct_change'] > 0.5) |  # 50% increase
            (demand_changes['demand_pct_change'] < -0.5)   # 50% decrease
        ]

        if not spike_days.empty:
            print(f"Found {len(spike_days)} days with extreme demand changes (>50% or <-50%)")

            # Get the original rows for these days
            for _, row in spike_days.iterrows():
                city = row['city']
                date = row['date']
                spike_rows = df[(df['city'] == city) & (df['date'] == date)]
                anomalies = pd.concat([anomalies, spike_rows])

    # Remove duplicate rows
    anomalies = anomalies.drop_duplicates()

    print(f"\nTotal anomalies found: {len(anomalies)}")

    return anomalies

# Detect anomalies
anomalies = detect_anomalies(merged_df, daily_agg)

print("\nSample of detected anomalies:")
print(anomalies.head())

# 8. HANDLE ANOMALIES
print("\n8. HANDLING ANOMALIES")
print("-" * 80)

def handle_anomalies(df, anomalies):
    """
    Handle detected anomalies in the dataset
    """
    print("Handling anomalies...")

    if anomalies.empty:
        print("No anomalies to handle.")
        return df

    # Create a copy of the dataframe
    df_cleaned = df.copy()

    # Identify anomalies by index
    anomaly_indices = anomalies.index

    # Strategy for handling anomalies:
    # 1. For temperature, humidity, and wind speed: impute with median
    for col in ['temperature', 'humidity', 'windSpeed']:
        if col in df.columns:
            # Identify anomalies in this column
            col_anomalies = anomalies[
                (anomalies[col] < df[col].quantile(0.01)) |
                (anomalies[col] > df[col].quantile(0.99))
            ].index

            if len(col_anomalies) > 0:
                print(f"Imputing {len(col_anomalies)} anomalies in {col}")

                # Replace with city-specific median
                for idx in col_anomalies:
                    if idx in df_cleaned.index:
                        city = df_cleaned.at[idx, 'city']
                        city_median = df_cleaned[df_cleaned['city'] == city][col].median()
                        df_cleaned.at[idx, col] = city_median

    # 2. For demand: impute with historical patterns
    if 'demand' in df.columns:
        # Identify demand anomalies
        demand_anomalies = anomalies[
            (anomalies['demand'] < df['demand'].quantile(0.01)) |
            (anomalies['demand'] > df['demand'].quantile(0.99))
        ].index

        if len(demand_anomalies) > 0:
            print(f"Imputing {len(demand_anomalies)} anomalies in demand")

            # Replace with similar day/hour average
            for idx in demand_anomalies:
                if idx in df_cleaned.index:
                    # Get city, hour, day of week
                    city = df_cleaned.at[idx, 'city']
                    hour = df_cleaned.at[idx, 'hour']
                    day_of_week = df_cleaned.at[idx, 'day_of_week']

                    # Find similar days/hours
                    similar_demand = df_cleaned[
                        (df_cleaned['city'] == city) &
                        (df_cleaned['hour'] == hour) &
                        (df_cleaned['day_of_week'] == day_of_week) &
                        (~df_cleaned.index.isin(anomaly_indices))
                    ]['demand'].mean()

                    # If we found similar days, use their average
                    if not pd.isna(similar_demand):
                        df_cleaned.at[idx, 'demand'] = similar_demand
                    else:
                        # Otherwise, use city median
                        city_median = df_cleaned[df_cleaned['city'] == city]['demand'].median()
                        df_cleaned.at[idx, 'demand'] = city_median

    return df_cleaned

# Handle anomalies
merged_df_cleaned = handle_anomalies(merged_df, anomalies)

print("\nMissing values after handling anomalies:")
print(merged_df_cleaned.isnull().sum()[merged_df_cleaned.isnull().sum() > 0])

# 9. SUMMARY STATISTICS
print("\n9. SUMMARY STATISTICS")
print("-" * 80)

def calculate_summary_statistics(df):
    """
    Calculate and display summary statistics for the dataset
    """
    print("Calculating summary statistics...")

    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    # Statistics by city
    if 'city' in df.columns and 'demand' in df.columns:
        print("\nDemand statistics by city:")
        print(df.groupby('city')['demand'].describe())

    # Statistics by season
    if 'season' in df.columns and 'demand' in df.columns:
        print("\nDemand statistics by season:")
        print(df.groupby('season')['demand'].describe())

    # Statistics by day of week
    if 'day_of_week' in df.columns and 'demand' in df.columns:
        print("\nDemand statistics by day of week:")
        day_stats = df.groupby('day_of_week')['demand'].describe()
        day_stats.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(day_stats)

    # Statistics by hour
    if 'hour' in df.columns and 'demand' in df.columns:
        print("\nDemand statistics by hour:")
        print(df.groupby('hour')['demand'].describe())

    return df

# Calculate summary statistics
merged_df_cleaned = calculate_summary_statistics(merged_df_cleaned)

# 10. SAVE PROCESSED DATA
print("\n10. SAVING PROCESSED DATA")
print("-" * 80)

def save_processed_data(df, daily_agg, weekly_agg):
    """
    Save processed data to CSV files
    """
    print("Saving processed data to CSV files...")

    # Save the main processed dataset
    df.to_csv('processed_energy_data.csv', index=False)
    print("Saved main dataset to processed_energy_data.csv")

    # Save daily aggregations
    daily_agg.to_csv('daily_energy_stats.csv', index=False)
    print("Saved daily aggregations to daily_energy_stats.csv")

    # Save weekly aggregations
    weekly_agg.to_csv('weekly_energy_stats.csv', index=False)
    print("Saved weekly aggregations to weekly_energy_stats.csv")

    return df

# Save processed data
merged_df_cleaned = save_processed_data(merged_df_cleaned, daily_agg, weekly_agg)

print("-" * 80)
print("DATA PREPROCESSING COMPLETE!")
print("-" * 80)
print(f"Total records in final dataset: {len(merged_df_cleaned)}")
print(f"Number of cities: {merged_df_cleaned['city'].nunique()}")
print(f"Date range: {merged_df_cleaned['local_time'].min()} to {merged_df_cleaned['local_time'].max()}")
print(f"Final dataset columns: {merged_df_cleaned.columns.tolist()}")
print("-" * 80)