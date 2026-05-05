#!/usr/bin/env python3
"""
Join Prediction Data with Station Coordinates

This script joins the neural network prediction data with station latitude/longitude
data to create the final enriched predictions file for the Streamlit application.

Input files:
- python notebooks/subway_delay_predictions.csv: Neural network predictions
- src/routes/resources/Station-lat-long - all-stations.csv: Station coordinates

Output file:
- src/routes/resources/enriched_predictions.csv: Final enriched predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_prediction_data():
    """Load the neural network prediction data"""
    print("Loading neural network predictions...")
    
    # Load the prediction data
    predictions_path = "python notebooks/subway_delay_predictions.csv"
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")
    
    # Read the predictions file
    predictions_df = pd.read_csv(predictions_path)
    
    print(f"Loaded {len(predictions_df):,} prediction records")
    print(f"Columns: {list(predictions_df.columns)}")
    print(f"Sample data:")
    print(predictions_df.head())
    
    return predictions_df

def load_station_coordinates():
    """Load station latitude/longitude data"""
    print("\nLoading station coordinates...")
    
    # Load the station coordinates
    coords_path = "src/routes/resources/Station-lat-long - all-stations.csv"
    
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Station coordinates file not found: {coords_path}")
    
    # Read the coordinates file
    coords_df = pd.read_csv(coords_path)
    
    print(f"Loaded {len(coords_df):,} station coordinates")
    print(f"Columns: {list(coords_df.columns)}")
    print(f"Sample data:")
    print(coords_df.head())
    
    return coords_df

def clean_station_names(df, column_name):
    """Clean station names for better matching"""
    # Remove common suffixes and standardize names
    df[column_name] = df[column_name].str.replace(' STATION', '', case=False)
    df[column_name] = df[column_name].str.replace(' ST', '', case=False)
    df[column_name] = df[column_name].str.strip()

    # Normalize dots and abbreviations so predictions match coords file
    name_map = {
        'ST GEORGE':               'ST. GEORGE',
        'ST CLAIR':                'ST. CLAIR',
        'ST CLAIR WEST':           'ST. CLAIR WEST',
        'ST PATRICK':              'ST. PATRICK',
        'ST ANDREW':               'ST. ANDREW',
        'NORTH YORK CTR':          'NORTH YORK CENTRE',
        'VAUGHAN MC':              'VAUGHAN METROPOLITAN CENTRE',
    }
    df[column_name] = df[column_name].replace(name_map)
    return df

def extract_hour_from_time(time_hhmm):
    """Extract hour from time string (HH:MM format)"""
    try:
        return int(time_hhmm.split(':')[0])
    except:
        return 0

def create_likelihood_from_severity_and_length(severity, length):
    """Create likelihood score from severity and length"""
    # Convert severity to numeric score
    severity_scores = {
        'Minimal': 0.1,
        'Minor': 0.3,
        'Moderate': 0.6,
        'Severe': 0.9
    }
    
    base_score = severity_scores.get(severity, 0.3)
    
    # Adjust based on delay length (normalize to 0-1 range)
    # Assuming max delay length is around 30 minutes
    length_factor = min(length / 30.0, 1.0)
    
    # Combine severity and length
    likelihood = base_score * 0.7 + length_factor * 0.3
    
    return min(likelihood, 1.0)

def process_predictions(predictions_df, coords_df):
    """Process and join prediction data with coordinates"""
    print("\nProcessing predictions...")
    
    # Clean station names in both dataframes
    predictions_df = clean_station_names(predictions_df, 'station_name')
    coords_df = clean_station_names(coords_df, 'station')
    
    # Extract hour from time_hhmm for additional analysis
    predictions_df['hour'] = predictions_df['time_hhmm'].apply(extract_hour_from_time)
    
    # Create likelihood score from severity and length
    predictions_df['likelihood_of_delay'] = predictions_df.apply(
        lambda row: create_likelihood_from_severity_and_length(
            row['delay_severity'], row['delay_length']
        ), axis=1
    )
    
    # Select all relevant columns for the final output
    output_columns = [
        'station_name',           # Station name
        'month',                  # Month (1-12)
        'day_of_week',           # Day of week (0=Monday, 6=Sunday)
        'is_weekend',            # Weekend flag (0 or 1)
        'time_norm',             # Normalized time (0.0-1.0)
        'time_hhmm',             # Time in HH:MM format
        'hour',                  # Extracted hour (0-23)
        'delay_severity',        # Delay severity (Minimal, Minor, Moderate, Severe)
        'delay_length',          # Delay length in minutes
        'likelihood_of_delay'    # Calculated likelihood score
    ]
    
    processed_df = predictions_df[output_columns].copy()
    
    # Rename station_name to station for joining
    processed_df = processed_df.rename(columns={'station_name': 'station'})
    
    print(f"Processed {len(processed_df):,} prediction records")
    print(f"Columns: {list(processed_df.columns)}")
    print(f"Month range: {processed_df['month'].min()} - {processed_df['month'].max()}")
    print(f"Day of week range: {processed_df['day_of_week'].min()} - {processed_df['day_of_week'].max()}")
    print(f"Hour range: {processed_df['hour'].min()} - {processed_df['hour'].max()}")
    print(f"Likelihood range: {processed_df['likelihood_of_delay'].min():.3f} - {processed_df['likelihood_of_delay'].max():.3f}")
    
    return processed_df

def join_with_coordinates(predictions_df, coords_df):
    """Join predictions with station coordinates"""
    print("\nJoining with station coordinates...")
    
    # Perform the join
    joined_df = predictions_df.merge(
        coords_df,
        on='station',
        how='left'
    )
    
    # Check for missing coordinates
    missing_coords = joined_df[joined_df['latitude'].isna() | joined_df['longitude'].isna()]
    if len(missing_coords) > 0:
        print(f"Warning: {len(missing_coords)} records have missing coordinates")
        print("Stations with missing coordinates:")
        print(missing_coords['station'].unique())
    
    # Remove records with missing coordinates
    joined_df = joined_df.dropna(subset=['latitude', 'longitude'])
    
    print(f"Final joined dataset: {len(joined_df):,} records")
    print(f"Unique stations: {joined_df['station'].nunique()}")
    print(f"Final columns: {list(joined_df.columns)}")
    
    return joined_df

def create_sample_data(joined_df, sample_size=1000):
    """Create a smaller sample for testing"""
    print(f"\nCreating sample dataset with {sample_size} records...")
    
    # Take a random sample
    sample_df = joined_df.sample(n=min(sample_size, len(joined_df)), random_state=42)
    
    return sample_df

def save_enriched_predictions(joined_df, output_path):
    """Save the enriched predictions to CSV"""
    print(f"\nSaving enriched predictions to {output_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    joined_df.to_csv(output_path, index=False)
    
    print(f"Successfully saved {len(joined_df):,} records")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    """Main function to process and join the data"""
    print("🚇 TTC Subway Delay Predictions - Data Joining Script")
    print("=" * 60)
    
    try:
        # Load data
        predictions_df = load_prediction_data()
        coords_df = load_station_coordinates()
        
        # Process predictions
        processed_df = process_predictions(predictions_df, coords_df)
        
        # Join with coordinates
        joined_df = join_with_coordinates(processed_df, coords_df)
        
        # Create sample for testing (smaller file)
        sample_df = create_sample_data(joined_df, sample_size=1000)
        
        # Save both full and sample datasets
        output_path = "src/routes/resources/enriched_predictions.csv"
        sample_path = "src/routes/resources/enriched_predictions_sample.csv"
        
        save_enriched_predictions(sample_df, output_path)
        save_enriched_predictions(joined_df, "src/routes/resources/enriched_predictions_full.csv")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("📊 SUMMARY STATISTICS")
        print("=" * 60)
        
        print(f"Total prediction records: {len(joined_df):,}")
        print(f"Unique stations: {joined_df['station'].nunique()}")
        print(f"Month range: {joined_df['month'].min()} - {joined_df['month'].max()}")
        print(f"Day of week range: {joined_df['day_of_week'].min()} - {joined_df['day_of_week'].max()}")
        print(f"Hour range: {joined_df['hour'].min()} - {joined_df['hour'].max()}")
        print(f"Average likelihood: {joined_df['likelihood_of_delay'].mean():.3f}")
        print(f"Likelihood range: {joined_df['likelihood_of_delay'].min():.3f} - {joined_df['likelihood_of_delay'].max():.3f}")
        
        # Station statistics
        station_stats = joined_df.groupby('station')['likelihood_of_delay'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(f"\nTop 5 highest risk stations:")
        print(station_stats.head())
        
        # Temporal statistics
        print(f"\nTemporal Analysis:")
        print(f"Weekend vs Weekday:")
        weekend_stats = joined_df.groupby('is_weekend')['likelihood_of_delay'].mean()
        print(f"  Weekend (1): {weekend_stats.get(1, 0):.3f}")
        print(f"  Weekday (0): {weekend_stats.get(0, 0):.3f}")
        
        print(f"\n✅ Data processing completed successfully!")
        print(f"📁 Output files:")
        print(f"   - {output_path} (sample for testing)")
        print(f"   - src/routes/resources/enriched_predictions_full.csv (full dataset)")
        print(f"📊 Final dataset includes all temporal features: month, day_of_week, is_weekend, time_norm, time_hhmm, hour, delay_severity, delay_length, likelihood_of_delay")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
