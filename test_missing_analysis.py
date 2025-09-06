#!/usr/bin/env python3
"""
Test script to demonstrate the missing data analysis capabilities.
Shows detailed statistics and recommendations.
"""

import pandas as pd
from missing_data_analysis import analyze_missing_data, MissingDataAnalyzer

def detailed_missing_analysis_demo():
    """Demonstrate detailed missing data analysis capabilities."""
    
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Run comprehensive analysis
    report = analyze_missing_data(df, output_dir='missing_data_plots')
    
    print("\n" + "="*60)
    print("DETAILED MISSING DATA STATISTICS")
    print("="*60)
    
    # Show detailed statistics
    analyzer = MissingDataAnalyzer(df)
    stats = analyzer.calculate_missing_statistics()
    
    print("\nColumns with Missing Data:")
    missing_cols = stats[stats['Missing_Count'] > 0]
    for _, row in missing_cols.iterrows():
        print(f"  • {row['Column']:<20}: {row['Missing_Count']:>5} missing ({row['Missing_Percentage']:>5.1f}%) - {row['Severity_Level']}")
    
    # Show pattern analysis
    patterns = analyzer.identify_missing_patterns()
    print(f"\nMissing Data Pattern Analysis:")
    print(f"  • Rows with no missing data: {patterns['row_analysis']['rows_with_no_missing']:,}")
    print(f"  • Rows with some missing data: {patterns['row_analysis']['rows_with_some_missing']:,}")
    print(f"  • Average missing values per row: {patterns['row_analysis']['avg_missing_per_row']:.2f}")
    
    if patterns['high_correlation_pairs']:
        print(f"\nHighly Correlated Missing Value Patterns:")
        for pair in patterns['high_correlation_pairs']:
            print(f"  • {pair['Column1']} ↔ {pair['Column2']}: {pair['Correlation']}")
    
    # Show placeholder analysis highlights
    placeholder_analysis = analyzer.detect_placeholder_values()
    print(f"\nPotential Placeholder Value Issues:")
    for col, analysis in placeholder_analysis.items():
        if analysis['suspected_placeholders']:
            print(f"  • {col}: {'; '.join(analysis['suspected_placeholders'])}")
    
    print(f"\n📁 Visualizations saved to: missing_data_plots/")
    print("   • missing_data_bar_chart.png - Count of missing values by column")
    print("   • missing_data_heatmap.png - Visual pattern of missing data")
    print("   • missing_correlation_heatmap.png - Correlation between missing values")
    print("   • missing_severity_pie_chart.png - Distribution of severity levels")
    print("   • missing_per_row_histogram.png - Missing values per row distribution")
    
    return report

if __name__ == "__main__":
    report = detailed_missing_analysis_demo()