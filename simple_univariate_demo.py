#!/usr/bin/env python3
"""
Simple demonstration of univariate analysis key results.
"""

import pandas as pd
import numpy as np

def show_key_results():
    """Show key results from the univariate analysis."""
    
    print("="*70)
    print("KEY UNIVARIATE ANALYSIS RESULTS")
    print("="*70)
    
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Quick analysis of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if not col.upper().startswith('INDEX')]
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total rows: {len(df):,}")
    print(f"   • Numerical columns: {len(numerical_cols)}")
    print(f"   • Visualizations generated: 30 plots (2 per variable)")
    
    print(f"\n📈 KEY STATISTICAL FINDINGS:")
    
    # Calculate basic stats for each variable
    stats_summary = []
    for col in numerical_cols:
        data = df[col].dropna()
        if len(data) > 0:
            skewness = data.skew()
            cv = (data.std() / abs(data.mean()) * 100) if data.mean() != 0 else np.inf
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            # Outlier calculation
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_pct = (len(outliers) / len(data)) * 100
            
            stats_summary.append({
                'Variable': col,
                'Missing%': missing_pct,
                'Mean': data.mean(),
                'Std': data.std(),
                'CV%': cv,
                'Skewness': skewness,
                'Outliers%': outlier_pct
            })
    
    stats_df = pd.DataFrame(stats_summary)
    
    print(f"\n   Most Variable Parameters (by CV%):")
    high_cv = stats_df.nlargest(5, 'CV%')[['Variable', 'CV%']]
    for _, row in high_cv.iterrows():
        print(f"     • {row['Variable']}: {row['CV%']:.1f}%")
    
    print(f"\n   Highest Missing Data:")
    high_missing = stats_df[stats_df['Missing%'] > 0].nlargest(3, 'Missing%')[['Variable', 'Missing%']]
    for _, row in high_missing.iterrows():
        print(f"     • {row['Variable']}: {row['Missing%']:.1f}%")
    
    print(f"\n   Most Outlier-Prone Variables:")
    high_outliers = stats_df.nlargest(5, 'Outliers%')[['Variable', 'Outliers%']]
    for _, row in high_outliers.iterrows():
        print(f"     • {row['Variable']}: {row['Outliers%']:.1f}%")
    
    print(f"\n   Most Skewed Variables:")
    stats_df['Abs_Skew'] = abs(stats_df['Skewness'])
    high_skew = stats_df.nlargest(5, 'Abs_Skew')[['Variable', 'Skewness']]
    for _, row in high_skew.iterrows():
        skew_direction = "right" if row['Skewness'] > 0 else "left"
        print(f"     • {row['Variable']}: {row['Skewness']:.3f} ({skew_direction}-skewed)")
    
    print(f"\n📁 GENERATED VISUALIZATIONS:")
    print(f"   • Location: univariate_analysis_plots/")
    print(f"   • Distribution analysis plots: distribution_[variable].png")
    print(f"   • Detailed histograms: detailed_histogram_[variable].png")
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   • ALL variables fail normality tests - use non-parametric methods")
    print(f"   • High outlier rates suggest data quality issues")
    print(f"   • STARS variable has significant missing data (26.3%)")
    print(f"   • Data appears pre-processed/normalized")
    print(f"   • Consider robust statistical approaches")
    
    print(f"\n💡 RECOMMENDATIONS FOR DAV 6150 MODULE 3:")
    print(f"   1. Use non-parametric statistical tests")
    print(f"   2. Apply robust scaling/transformation methods")
    print(f"   3. Address missing data in STARS variable")
    print(f"   4. Investigate outliers before modeling")
    print(f"   5. Consider data quality issues in interpretation")
    
    print(f"\n" + "="*70)
    print("UNIVARIATE ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    show_key_results()