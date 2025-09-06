#!/usr/bin/env python3
"""
Detailed examination of correlation analysis results.
Focus on wine chemistry relationships and TARGET correlations.
"""

import pandas as pd
import numpy as np
from correlation_analysis import comprehensive_correlation_analysis

def examine_correlation_findings():
    """Examine detailed correlation analysis findings."""
    
    print("="*70)
    print("DETAILED CORRELATION ANALYSIS EXAMINATION")
    print("="*70)
    
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Run correlation analysis
    results = comprehensive_correlation_analysis(df, target_column='TARGET')
    
    print("\n🔍 TARGET VARIABLE CORRELATION ANALYSIS:")
    print("-" * 50)
    
    target_analysis = results['target_analysis']
    if 'target_correlations' in target_analysis:
        target_corr = target_analysis['target_correlations']
        
        print("Top 10 Variables Correlated with Wine Sales (TARGET):")
        top_10 = target_corr.head(10)
        
        for _, row in top_10.iterrows():
            significance = "***" if row['Pearson_P_Value'] < 0.001 else \
                         "**" if row['Pearson_P_Value'] < 0.01 else \
                         "*" if row['Pearson_P_Value'] < 0.05 else ""
            
            relationship = "↑" if row['Pearson_Correlation'] > 0 else "↓"
            
            print(f"  {relationship} {row['Variable']:<18}: r = {row['Pearson_Correlation']:>6.3f}{significance:>3} " +
                  f"(p = {row['Pearson_P_Value']:.4f})")
        
        print("\n🍷 WINE CHEMISTRY INSIGHTS:")
        print("-" * 50)
        
        # Group variables by wine chemistry categories
        acidity_vars = ['FixedAcidity', 'VolatileAcidity', 'CitricAcid', 'pH', 'AcidIndex']
        sugar_alcohol = ['ResidualSugar', 'Alcohol', 'Density']
        sulfur_vars = ['FreeSulfurDioxide', 'TotalSulfurDioxide', 'Sulphates']
        other_chem = ['Chlorides']
        quality_vars = ['STARS', 'LabelAppeal']
        
        def analyze_category(category_name, variables):
            print(f"\n  {category_name}:")
            for var in variables:
                if var in target_corr['Variable'].values:
                    row = target_corr[target_corr['Variable'] == var].iloc[0]
                    significance = "***" if row['Pearson_P_Value'] < 0.001 else \
                                 "**" if row['Pearson_P_Value'] < 0.01 else \
                                 "*" if row['Pearson_P_Value'] < 0.05 else ""
                    
                    impact = "increases" if row['Pearson_Correlation'] > 0 else "decreases"
                    strength = "strongly" if abs(row['Pearson_Correlation']) > 0.3 else \
                             "moderately" if abs(row['Pearson_Correlation']) > 0.1 else "weakly"
                    
                    print(f"    • {var}: {strength} {impact} sales " +
                          f"(r = {row['Pearson_Correlation']:.3f}{significance})")
        
        analyze_category("Acidity & pH", acidity_vars)
        analyze_category("Sugar & Alcohol", sugar_alcohol)  
        analyze_category("Sulfur Compounds", sulfur_vars)
        analyze_category("Other Chemistry", other_chem)
        analyze_category("Quality Ratings", quality_vars)
    
    print(f"\n📊 MULTICOLLINEARITY ANALYSIS:")
    print("-" * 50)
    
    vif_results = results['multicollinearity_analysis']
    if 'vif_results' in vif_results:
        vif_df = vif_results['vif_results']
        
        print("Variables with Multicollinearity Issues:")
        high_vif = vif_df[vif_df['VIF'] > 5].head(10)
        
        for _, row in high_vif.iterrows():
            if not np.isnan(row['VIF']) and row['VIF'] != np.inf:
                risk_level = "🔴 Critical" if row['VIF'] > 10 else \
                           "🟡 Moderate" if row['VIF'] > 5 else "🟢 Low"
                print(f"  {risk_level}: {row['Variable']:<18} VIF = {row['VIF']:>6.2f} ({row['Interpretation']})")
    
    print(f"\n🔗 STRONG CORRELATIONS BETWEEN CHEMISTRY PARAMETERS:")
    print("-" * 50)
    
    # Extract strong correlations from the correlation matrix
    corr_matrix = results['correlation_summary']['pearson_correlations']
    
    strong_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            
            if abs(correlation) > 0.4 and not pd.isna(correlation):
                strong_pairs.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': correlation,
                    'abs_corr': abs(correlation)
                })
    
    # Sort by absolute correlation
    strong_pairs = sorted(strong_pairs, key=lambda x: x['abs_corr'], reverse=True)
    
    print("Strongest Inter-Variable Correlations:")
    for i, pair in enumerate(strong_pairs[:10]):
        relationship = "positively" if pair['correlation'] > 0 else "negatively"
        strength = "very strongly" if pair['abs_corr'] > 0.7 else \
                  "strongly" if pair['abs_corr'] > 0.5 else "moderately"
        
        print(f"  {i+1:2d}. {pair['var1']} ↔ {pair['var2']}")
        print(f"      {strength} {relationship} correlated (r = {pair['correlation']:.3f})")
    
    print(f"\n💡 BUSINESS INSIGHTS FOR WINE SALES:")
    print("-" * 50)
    
    if 'target_correlations' in target_analysis:
        # Find most actionable insights
        top_positive = target_corr[target_corr['Pearson_Correlation'] > 0].head(3)
        top_negative = target_corr[target_corr['Pearson_Correlation'] < 0].head(3)
        
        print("To INCREASE wine sales, focus on:")
        for _, row in top_positive.iterrows():
            if row['Pearson_Significant']:
                action = "higher" if row['Pearson_Correlation'] > 0 else "lower"
                print(f"  • {action} {row['Variable']} (correlation: {row['Pearson_Correlation']:.3f})")
        
        print("\nVariables that may HURT sales if increased:")
        for _, row in top_negative.iterrows():
            if row['Pearson_Significant'] and row['Pearson_Correlation'] < -0.1:
                print(f"  • {row['Variable']} (correlation: {row['Pearson_Correlation']:.3f})")
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE SUMMARY:")
    print("-" * 50)
    
    if 'target_correlations' in target_analysis:
        sig_count = target_corr['Pearson_Significant'].sum()
        total_vars = len(target_corr)
        
        print(f"• Significant correlations with TARGET: {sig_count}/{total_vars} ({sig_count/total_vars*100:.1f}%)")
        
        # Categorize significance levels
        very_sig = (target_corr['Pearson_P_Value'] < 0.001).sum()
        mod_sig = ((target_corr['Pearson_P_Value'] >= 0.001) & (target_corr['Pearson_P_Value'] < 0.01)).sum()
        weak_sig = ((target_corr['Pearson_P_Value'] >= 0.01) & (target_corr['Pearson_P_Value'] < 0.05)).sum()
        
        print(f"• Highly significant (p < 0.001): {very_sig} variables")
        print(f"• Moderately significant (0.001 ≤ p < 0.01): {mod_sig} variables")  
        print(f"• Weakly significant (0.01 ≤ p < 0.05): {weak_sig} variables")
    
    print(f"\n📁 GENERATED VISUALIZATIONS:")
    print(f"   • correlation_heatmaps_combined.png - Side-by-side Pearson & Spearman")
    print(f"   • detailed_pearson_correlations.png - With significance indicators")
    print(f"   • target_correlations_analysis.png - Wine sales correlations")
    print(f"   • vif_multicollinearity_analysis.png - Multicollinearity assessment")
    print(f"   • scatter_plot_matrix.png - Pairwise relationships")
    print(f"   • correlation_network.png - Network of strong correlations")
    
    print(f"\n🎯 DAV 6150 MODULE 3 RECOMMENDATIONS:")
    print("-" * 50)
    print("1. STARS rating is the strongest predictor of wine sales - focus on quality")
    print("2. Consider multicollinearity when building predictive models")
    print("3. Wine chemistry has moderate but significant impact on sales")
    print("4. AcidIndex shows negative correlation - balance acidity carefully")
    print("5. Use both chemical properties and quality ratings for sales prediction")
    
    print("\n" + "="*70)
    print("CORRELATION EXAMINATION COMPLETE")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = examine_correlation_findings()