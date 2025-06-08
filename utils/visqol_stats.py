"""
Calculate statistics for Visqol scores from audio mark evaluation results.

Usage:
    python visqol_stats.py <input_file>

Example:
    python visqol_stats.py ../eval_results/8khz_10hrs_125epochs.txt
"""

import argparse
import os
import pandas as pd
import numpy as np

def calculate_visqol_stats(file_path):
    """
    Calculate statistics for Visqol scores from the given file.
    
    Args:
        file_path (str): Path to the input text file containing evaluation results
        
    Returns:
        dict: Dictionary containing the calculated statistics
    """
    try:
        # Read the data file
        df = pd.read_csv(file_path, sep=',\s*', engine='python')
        
        # Clean up column names (remove any leading/trailing whitespace)
        df.columns = df.columns.str.strip()
        
        # Find the Visqol score column (case-insensitive)
        visqol_col = None
        for col in df.columns:
            if 'visqol' in col.lower():
                visqol_col = col
                break
                
        if visqol_col is None:
            raise ValueError("No Visqol score column found in the input file")
            
        # Calculate statistics
        stats = {
            'file': os.path.basename(file_path),
            'count': len(df),
            'max': df[visqol_col].max(),
            'min': df[visqol_col].min(),
            'mean': df[visqol_col].mean(),
            'median': df[visqol_col].median(),
            'std': df[visqol_col].std()
        }
        
        return stats
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Calculate Visqol score statistics from audio mark evaluation results')
    parser.add_argument('input_file', help='Path to the input text file with evaluation results')
    args = parser.parse_args()
    
    stats = calculate_visqol_stats(args.input_file)
    
    if stats:
        def truncate(num, digits=3):
            # Truncate to specified number of decimal places
            multiplier = 10 ** digits
            return int(num * multiplier) / multiplier
            
        print(f"\nStatistics for {stats['file']}:")
        print(f"  Samples: {stats['count']}")
        print(f"  Max Visqol: {truncate(stats['max']):.3f}")
        print(f"  Min Visqol: {truncate(stats['min']):.3f}")
        print(f"  Mean Visqol: {truncate(stats['mean']):.3f}")
        print(f"  Median Visqol: {truncate(stats['median']):.3f}")
        print(f"  Std Dev: {truncate(stats['std']):.3f}")

if __name__ == "__main__":
    main()
