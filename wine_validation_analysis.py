#!/usr/bin/env python3
"""
Detailed analysis of wine chemistry validation results.
Examines specific validation issues and provides insights.
"""

import pandas as pd
import numpy as np
from wine_chemistry_validator import validate_wine_chemistry

def analyze_validation_results():
    """Perform detailed analysis of wine chemistry validation results."""
    
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Run validation
    validation_report = validate_wine_chemistry(df)
    
    print("\n" + "="*70)
    print("DETAILED VALIDATION ANALYSIS")
    print("="*70)
    
    # Get flagged dataset
    df_flagged = validation_report['flagged_dataset']
    
    # Analysis 1: Distribution of quality issues
    print("\n📊 QUALITY ISSUE DISTRIBUTION:")
    print("-" * 50)
    quality_dist = df_flagged['QF_OVERALL_QUALITY'].value_counts()
    for quality, count in quality_dist.items():
        percentage = (count / len(df_flagged)) * 100
        print(f"  • {quality:<12}: {count:>6,} rows ({percentage:>5.1f}%)")
    
    # Analysis 2: Parameter-specific impossible values
    print("\n🔍 PARAMETER-SPECIFIC ANALYSIS:")
    print("-" * 50)
    
    param_ranges = {
        'pH': (2.9, 4.0),
        'Alcohol': (8.0, 16.0), 
        'FixedAcidity': (4.0, 12.0),
        'VolatileAcidity': (0.1, 1.2),
        'ResidualSugar': (0.5, 150.0),
        'Density': (0.985, 1.005)
    }
    
    for param, (min_val, max_val) in param_ranges.items():
        if param in df.columns:
            param_data = df[param].dropna()
            below_min = (param_data < min_val).sum()
            above_max = (param_data > max_val).sum()
            min_actual = param_data.min()
            max_actual = param_data.max()
            
            print(f"\n  {param}:")
            print(f"    Expected range: {min_val} - {max_val}")
            print(f"    Actual range:   {min_actual:.3f} - {max_actual:.3f}")
            print(f"    Below minimum:  {below_min:,} values")
            print(f"    Above maximum:  {above_max:,} values")
            
            # Show some example problematic values
            if below_min > 0:
                worst_below = param_data[param_data < min_val].nsmallest(3)
                print(f"    Worst low values: {worst_below.tolist()}")
            if above_max > 0:
                worst_above = param_data[param_data > max_val].nlargest(3)
                print(f"    Worst high values: {worst_above.tolist()}")
    
    # Analysis 3: Logical relationship violations
    print("\n🔗 LOGICAL RELATIONSHIP VIOLATIONS:")
    print("-" * 50)
    
    # Total SO2 vs Free SO2 violations
    if 'TotalSulfurDioxide' in df.columns and 'FreeSulfurDioxide' in df.columns:
        violation_mask = (df['TotalSulfurDioxide'] < df['FreeSulfurDioxide']) & \
                        df['TotalSulfurDioxide'].notna() & df['FreeSulfurDioxide'].notna()
        violation_count = violation_mask.sum()
        print(f"  • Total SO2 < Free SO2: {violation_count:,} violations")
        
        if violation_count > 0:
            # Show examples
            examples = df[violation_mask][['TotalSulfurDioxide', 'FreeSulfurDioxide']].head(5)
            print("    Examples (Total, Free):")
            for idx, row in examples.iterrows():
                print(f"      Row {idx}: {row['TotalSulfurDioxide']:.1f}, {row['FreeSulfurDioxide']:.1f}")
    
    # Analysis 4: Missing data vs validation flags correlation
    print("\n📋 MISSING DATA vs QUALITY FLAGS:")
    print("-" * 50)
    
    # Count missing values per row
    missing_per_row = df.isnull().sum(axis=1)
    quality_with_missing = pd.DataFrame({
        'Missing_Count': missing_per_row,
        'Quality_Flag': df_flagged['QF_OVERALL_QUALITY']
    })
    
    missing_by_quality = quality_with_missing.groupby('Quality_Flag')['Missing_Count'].agg(['mean', 'count'])
    print("  Average missing values by quality level:")
    for quality in ['GOOD', 'QUESTIONABLE', 'POOR', 'CRITICAL']:
        if quality in missing_by_quality.index:
            avg_missing = missing_by_quality.loc[quality, 'mean']
            count = missing_by_quality.loc[quality, 'count']
            print(f"    {quality:<12}: {avg_missing:.2f} avg missing ({count:,} rows)")
    
    # Analysis 5: Recommendations for data cleaning
    print("\n💡 DATA CLEANING RECOMMENDATIONS:")
    print("-" * 50)
    
    impossible_count = (df_flagged['QF_IMPOSSIBLE']).sum()
    extreme_count = (df_flagged['QF_EXTREME_OUTLIER']).sum()
    logical_count = (df_flagged['QF_LOGICAL_VIOLATION']).sum()
    
    print(f"1. IMMEDIATE ACTION NEEDED:")
    print(f"   • {impossible_count:,} rows with impossible values - MUST be corrected")
    print(f"   • {logical_count:,} rows with logical violations - investigate data collection")
    
    print(f"\n2. REVIEW RECOMMENDED:")
    print(f"   • {extreme_count:,} rows with extreme outliers - verify measurements")
    
    print(f"\n3. DATA SOURCE INVESTIGATION:")
    print("   • High number of negative values in parameters that should be positive")
    print("   • Possible unit conversion errors or data encoding issues")
    print("   • Consider if data has been normalized/standardized already")
    
    # Analysis 6: Focus on specific problematic parameters
    print("\n🎯 MOST PROBLEMATIC PARAMETERS:")
    print("-" * 50)
    
    problem_params = []
    for param in ['FixedAcidity', 'VolatileAcidity', 'CitricAcid', 'ResidualSugar', 'Chlorides']:
        if param in df.columns:
            negative_count = (df[param] < 0).sum()
            if negative_count > 0:
                problem_params.append((param, negative_count))
    
    problem_params.sort(key=lambda x: x[1], reverse=True)
    
    for param, neg_count in problem_params[:5]:
        percentage = (neg_count / len(df)) * 100
        print(f"  • {param}: {neg_count:,} negative values ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return validation_report

if __name__ == "__main__":
    report = analyze_validation_results()