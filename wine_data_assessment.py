import pandas as pd
import numpy as np

def load_and_assess_data(file_path='M3_Data.csv'):
    """
    Load wine dataset and perform comprehensive initial data assessment.
    
    This function provides a complete overview of the dataset structure, quality,
    and potential issues that need to be addressed during analysis.
    
    Parameters:
    -----------
    file_path : str, default 'M3_Data.csv'
        Path to the CSV file containing the wine dataset
        
    Returns:
    --------
    pandas.DataFrame
        Loaded and initially assessed wine dataset
    """
    
    print("="*60)
    print("WINE DATASET INITIAL ASSESSMENT")
    print("="*60)
    
    # Step 1: Load the dataset with proper handling of potential encoding issues
    print("\n1. LOADING DATASET...")
    try:
        # Load with UTF-8-BOM encoding to handle the BOM character at the beginning
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✓ Dataset loaded successfully from: {file_path}")
    except UnicodeDecodeError:
        # Fallback to different encoding if UTF-8-BOM fails
        df = pd.read_csv(file_path, encoding='latin-1')
        print(f"✓ Dataset loaded with latin-1 encoding from: {file_path}")
    except FileNotFoundError:
        print(f"✗ Error: File '{file_path}' not found")
        return None
    
    # Step 2: Dataset shape and basic structure
    print("\n2. DATASET SHAPE AND STRUCTURE")
    print("-" * 40)
    print(f"Dataset dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Total data points: {df.size:,}")
    
    # Step 3: Column information and data types
    print("\n3. COLUMNS AND DATA TYPES")
    print("-" * 40)
    print("Column names and types:")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        print(f"{i:2d}. {col:<20} - {dtype}")
    
    # Step 4: Memory usage analysis
    print("\n4. MEMORY USAGE ANALYSIS")
    print("-" * 40)
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    print(f"Total memory usage: {total_memory / 1024 / 1024:.2f} MB")
    print("\nMemory usage by column:")
    for col, mem in memory_usage.items():
        if col != 'Index':  # Skip the index
            print(f"  {col:<20}: {mem / 1024:.2f} KB")
    
    # Step 5: First few rows inspection
    print("\n5. FIRST 5 ROWS")
    print("-" * 40)
    print(df.head())
    
    # Step 6: Basic dataset info
    print("\n6. DATASET INFO SUMMARY")
    print("-" * 40)
    print("Data types distribution:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Step 7: Missing values analysis
    print("\n7. MISSING VALUES ANALYSIS")
    print("-" * 40)
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    if missing_values.sum() == 0:
        print("✓ No missing values detected")
    else:
        print("Missing values found:")
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing %': missing_percentage.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        print(missing_df.to_string(index=False))
    
    # Step 8: Quick identification of potential issues
    print("\n8. POTENTIAL DATA QUALITY ISSUES")
    print("-" * 40)
    issues_found = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues_found.append(f"Duplicate rows: {duplicates}")
    
    # Check for columns with very high missing values (>50%)
    high_missing = missing_percentage[missing_percentage > 50]
    if len(high_missing) > 0:
        issues_found.append(f"Columns with >50% missing: {list(high_missing.index)}")
    
    # Check for potential outliers in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'INDEX':  # Skip index column
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))][col].count()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                issues_found.append(f"{col}: {outliers} potential outliers ({outliers/len(df)*100:.1f}%)")
    
    # Check for mixed data types in object columns
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric to see if it's mixed
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            if not numeric_conversion.isnull().all() and numeric_conversion.isnull().any():
                issues_found.append(f"{col}: Contains mixed data types")
    
    if issues_found:
        for issue in issues_found:
            print(f"⚠ {issue}")
    else:
        print("✓ No major data quality issues detected")
    
    # Step 9: Basic statistical summary for numeric columns
    print("\n9. NUMERIC COLUMNS STATISTICAL SUMMARY")
    print("-" * 40)
    numeric_summary = df.select_dtypes(include=[np.number]).describe()
    print(numeric_summary.round(3))
    
    print("\n" + "="*60)
    print("ASSESSMENT COMPLETE")
    print("="*60)
    
    return df

# Example usage and testing
if __name__ == "__main__":
    # Load and assess the wine dataset
    wine_data = load_and_assess_data()