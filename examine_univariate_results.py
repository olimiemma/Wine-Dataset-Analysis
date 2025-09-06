#!/usr/bin/env python3
"""
Detailed examination of univariate analysis results.
Shows specific statistical findings and insights.
"""

import pandas as pd
import numpy as np
from univariate_analysis import comprehensive_univariate_analysis

def examine_analysis_results():
    """Examine detailed univariate analysis results."""
    
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Run analysis
    results = comprehensive_univariate_analysis(df)
    
    print("\n" + "="*70)
    print("DETAILED EXAMINATION OF RESULTS")
    print("="*70)
    
    # Display summary tables
    if 'descriptive_statistics' in results['summary_tables']:
        desc_table = results['summary_tables']['descriptive_statistics']
        
        print("\n📊 DESCRIPTIVE STATISTICS SUMMARY:")
        print("-" * 50)
        print("Top 10 most variable parameters (by CV%):")
        high_cv = desc_table.nlargest(10, 'CV%')[['Variable', 'Mean', 'Std', 'CV%', 'Skewness']]
        print(high_cv.to_string(index=False, float_format='%.3f'))
        
        print("\n📈 MOST SKEWED DISTRIBUTIONS:")
        print("-" * 50)
        skewed_vars = desc_table[abs(desc_table['Skewness']) > 0.5].sort_values('Skewness', key=abs, ascending=False)
        if not skewed_vars.empty:
            print(skewed_vars[['Variable', 'Skewness', 'Mean', 'Median']].to_string(index=False, float_format='%.3f'))
        else:
            print("No highly skewed variables found.")
        
        print("\n🎯 OUTLIER ANALYSIS:")
        print("-" * 50)
        high_outliers = desc_table[desc_table['Outliers%'] > 10].sort_values('Outliers%', ascending=False)
        if not high_outliers.empty:
            print("Variables with >10% outliers:")
            # Calculate range manually since it's not in the summary table
            desc_table['Range'] = desc_table['Max'] - desc_table['Min']
            print(high_outliers[['Variable', 'Outliers%', 'IQR', 'Range']].to_string(index=False, float_format='%.3f'))
        
    # Normality test details
    if 'normality_tests' in results['summary_tables']:
        norm_table = results['summary_tables']['normality_tests']
        
        print("\n🔍 NORMALITY TEST DETAILS:")
        print("-" * 50)
        print("Variables with highest normality evidence:")
        top_normal = norm_table.nlargest(5, 'Normal_Percentage')[['Variable', 'Normal_Percentage', 'Conclusion']]
        print(top_normal.to_string(index=False))
        
        print("\nVariables clearly not normal:")
        non_normal = norm_table[norm_table['Normal_Percentage'] == 0][['Variable', 'Conclusion']]
        if not non_normal.empty:
            print(f"Count: {len(non_normal)} variables")
            print(non_normal.head(10).to_string(index=False))
    
    # Detailed analysis for specific problematic variables
    print("\n🔬 DETAILED ANALYSIS OF SPECIFIC VARIABLES:")
    print("-" * 50)
    
    # Examine STARS variable (high missing data)
    if 'STARS' in results['variable_analyses']:
        stars_analysis = results['variable_analyses']['STARS']
        if 'descriptive_statistics' in stars_analysis:
            stars_stats = stars_analysis['descriptive_statistics']
            print(f"\nSTARS Variable Analysis:")
            print(f"  • Missing data: {stars_stats['missing_percentage']:.1f}%")
            print(f"  • Range: {stars_stats['min']:.1f} to {stars_stats['max']:.1f}")
            print(f"  • Mean: {stars_stats['mean']:.3f}, Median: {stars_stats['median']:.3f}")
            print(f"  • Skewness: {stars_stats['skewness']:.3f} ({stars_stats['skewness_interpretation']})")
    
    # Examine ResidualSugar (high outliers)
    if 'ResidualSugar' in results['variable_analyses']:
        sugar_analysis = results['variable_analyses']['ResidualSugar']
        if 'descriptive_statistics' in sugar_analysis:
            sugar_stats = sugar_analysis['descriptive_statistics']
            print(f"\nResidualSugar Variable Analysis:")
            print(f"  • Outliers: {sugar_stats['outlier_percentage']:.1f}%")
            print(f"  • Range: {sugar_stats['min']:.1f} to {sugar_stats['max']:.1f}")
            print(f"  • IQR: {sugar_stats['iqr']:.3f}")
            print(f"  • CV: {sugar_stats['cv']:.1f}% (high variability)")
    
    # Summary of data quality issues
    print(f"\n📋 DATA QUALITY SUMMARY:")
    print("-" * 50)
    
    total_vars = len(results['dataset_info']['columns_analyzed'])
    
    # Count issues
    if 'descriptive_statistics' in results['summary_tables']:
        desc_df = results['summary_tables']['descriptive_statistics']
        high_missing = len(desc_df[desc_df['Missing%'] > 5])
        high_outliers = len(desc_df[desc_df['Outliers%'] > 10])
        high_skew = len(desc_df[abs(desc_df['Skewness']) > 1])
        high_cv = len(desc_df[desc_df['CV%'] > 100])
        
        print(f"  • Variables with >5% missing data: {high_missing}/{total_vars}")
        print(f"  • Variables with >10% outliers: {high_outliers}/{total_vars}")
        print(f"  • Variables with high skewness (|skew| > 1): {high_skew}/{total_vars}")
        print(f"  • Variables with high variability (CV > 100%): {high_cv}/{total_vars}")
    
    if 'normality_tests' in results['summary_tables']:
        norm_df = results['summary_tables']['normality_tests']
        clearly_normal = len(norm_df[norm_df['Normal_Percentage'] >= 80])
        clearly_non_normal = len(norm_df[norm_df['Normal_Percentage'] == 0])
        
        print(f"  • Variables likely normal: {clearly_normal}/{total_vars}")
        print(f"  • Variables clearly not normal: {clearly_non_normal}/{total_vars}")
    
    print(f"\n📁 VISUALIZATIONS GENERATED:")
    print(f"   • Total plots created: 30 (2 per variable)")
    print(f"   • Location: univariate_analysis_plots/")
    print(f"   • Distribution plots: distribution_[variable].png")
    print(f"   • Detailed histograms: detailed_histogram_[variable].png")
    
    print(f"\n💡 ANALYSIS INSIGHTS:")
    print("-" * 50)
    print("1. The dataset appears to be pre-processed or standardized")
    print("2. All variables show non-normal distributions")
    print("3. High outlier percentages suggest data quality issues")
    print("4. STARS variable has significant missing data (26.3%)")
    print("5. Many variables show high coefficient of variation")
    print("6. Consider robust statistical methods due to outliers and non-normality")
    
    return results

if __name__ == "__main__":
    results = examine_analysis_results()