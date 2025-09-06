#!/usr/bin/env python3
"""
Detailed examination of outlier analysis results.
Focus on understanding the high outlier rates and providing insights.
"""

import pandas as pd
import numpy as np
from outlier_analysis import comprehensive_outlier_analysis

def examine_outlier_findings():
    """Examine detailed outlier analysis findings."""
    
    print("="*70)
    print("DETAILED OUTLIER ANALYSIS EXAMINATION")
    print("="*70)
    
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Run outlier analysis
    results = comprehensive_outlier_analysis(df)
    
    print("\n🔍 UNDERSTANDING THE HIGH OUTLIER RATE (99.0%):")
    print("-" * 50)
    print("The extremely high outlier rate suggests:")
    print("1. Data appears to be pre-processed/normalized")
    print("2. Traditional outlier thresholds may not apply")
    print("3. Domain-specific validation reveals data quality issues")
    print("4. Multiple methods agree on problematic data points")
    
    print("\n📊 METHOD-BY-METHOD BREAKDOWN:")
    print("-" * 50)
    
    # Analyze individual method results
    individual_methods = results['individual_methods']
    
    method_summary = {}
    for variable, methods in individual_methods.items():
        if variable not in ['multivariate', 'domain_specific'] and isinstance(methods, dict):
            for method_name, method_result in methods.items():
                if method_name not in method_summary:
                    method_summary[method_name] = []
                
                if 'outlier_percentage' in method_result:
                    method_summary[method_name].append(method_result['outlier_percentage'])
    
    print("Average outlier detection rates by method:")
    for method, percentages in method_summary.items():
        avg_rate = np.mean(percentages)
        max_rate = np.max(percentages)
        min_rate = np.min(percentages)
        print(f"  • {method:<20}: {avg_rate:>5.1f}% avg (range: {min_rate:.1f}% - {max_rate:.1f}%)")
    
    print("\n🍷 DOMAIN-SPECIFIC ANALYSIS BREAKDOWN:")
    print("-" * 50)
    
    domain_results = results['individual_methods']['domain_specific']
    
    print(f"Wine Chemistry Violations:")
    print(f"  • Chemically impossible values: {len(domain_results['chemically_impossible']):,} data points")
    print(f"  • Extremely rare but possible: {len(domain_results['extremely_rare']):,} data points")
    print(f"  • Suspicious combinations: {len(domain_results['suspicious_combinations']):,} data points")
    
    # Examine parameter-specific violations
    print(f"\nParameter-specific violations:")
    if 'parameter_violations' in domain_results:
        for param, violations in domain_results['parameter_violations'].items():
            impossible = violations.get('impossible_count', 0)
            rare = violations.get('rare_count', 0)
            total_param_issues = impossible + rare
            
            print(f"  • {param:<18}: {total_param_issues:>4,} issues ({impossible:,} impossible, {rare:,} rare)")
    
    print("\n🎯 MULTIVARIATE OUTLIER FINDINGS:")
    print("-" * 50)
    
    multivariate_results = results['individual_methods']['multivariate']
    
    for method_name, method_result in multivariate_results.items():
        if 'outlier_count' in method_result:
            count = method_result['outlier_count']
            percentage = method_result['outlier_percentage']
            print(f"  • {method_name}: {count:,} outliers ({percentage:.1f}%)")
    
    print("\n🤝 CONSENSUS SCORING INSIGHTS:")
    print("-" * 50)
    
    consensus = results['consensus_analysis']
    
    print(f"Outlier Severity Distribution:")
    print(f"  • Severe outliers (immediate action):  {len(consensus['severe_outliers']):>4,} ({len(consensus['severe_outliers'])/len(df)*100:.1f}%)")
    print(f"  • Moderate outliers (review needed): {len(consensus['moderate_outliers']):>4,} ({len(consensus['moderate_outliers'])/len(df)*100:.1f}%)")
    print(f"  • Mild outliers (monitoring):        {len(consensus['mild_outliers']):>4,} ({len(consensus['mild_outliers'])/len(df)*100:.1f}%)")
    print(f"  • Normal data points:                {(len(df) - len(consensus['severe_outliers']) - len(consensus['moderate_outliers']) - len(consensus['mild_outliers'])):>4,} ({(len(df) - len(consensus['severe_outliers']) - len(consensus['moderate_outliers']) - len(consensus['mild_outliers']))/len(df)*100:.1f}%)")
    
    print(f"\nMethod Agreement Analysis:")
    method_counts = consensus['method_counts']
    agreement_stats = method_counts[method_counts > 0].value_counts().sort_index()
    
    for agreement_level, count in agreement_stats.items():
        print(f"  • {agreement_level} methods agree: {count:,} data points")
    
    print("\n💊 TREATMENT RECOMMENDATIONS ANALYSIS:")
    print("-" * 50)
    
    treatment_recs = results['treatment_recommendations']
    
    print("Immediate Actions Required:")
    for action in treatment_recs.get('immediate_action', []):
        print(f"  • {action}")
    
    print("\nReview Recommendations:")
    for rec in treatment_recs.get('review_recommended', []):
        print(f"  • {rec}")
    
    print("\nTreatment Strategies:")
    for strategy_type, strategies in treatment_recs.get('treatment_strategies', {}).items():
        print(f"  {strategy_type.replace('_', ' ').title()}:")
        for strategy in strategies:
            print(f"    - {strategy}")
    
    print("\n📈 VARIABLE-SPECIFIC ISSUES:")
    print("-" * 50)
    
    var_actions = treatment_recs.get('variable_specific_actions', {})
    if var_actions:
        print("Variables requiring special attention:")
        for variable, action_info in var_actions.items():
            print(f"  • {variable}:")
            print(f"    Issue: {action_info['issue']}")
            print(f"    Detected by: {action_info['detected_by']}")
            print(f"    Top recommendation: {action_info['recommendations'][0]}")
    else:
        print("No variables flagged for specific treatment actions.")
    
    print("\n🔄 BEFORE/AFTER TREATMENT ANALYSIS:")
    print("-" * 50)
    
    before_after = results['before_after_analysis']
    treatment_summary = before_after['treatment_summary']
    
    print(f"Treatment method applied: {treatment_summary['method']}")
    
    if 'rows_removed' in treatment_summary:
        print(f"Rows removed: {treatment_summary['rows_removed']:,}")
    
    # Show statistical improvements for key variables
    statistical_comparison = before_after['statistical_comparison']
    key_variables = ['TARGET', 'STARS', 'Alcohol', 'pH', 'FixedAcidity']
    
    print(f"\nStatistical changes after treatment (key variables):")
    for var in key_variables:
        if var in statistical_comparison:
            changes = statistical_comparison[var]['changes']
            print(f"  • {var}:")
            print(f"    Standard deviation reduction: {changes['std_reduction']:>6.3f}")
            print(f"    Range reduction: {changes['range_reduction']:>6.3f}")
            print(f"    Skewness improvement: {changes['skewness_improvement']:>6.3f}")
    
    print("\n🎯 MACHINE LEARNING IMPLICATIONS:")
    print("-" * 50)
    
    outlier_percentage = consensus['outlier_percentage']
    severe_percentage = len(consensus['severe_outliers']) / len(df) * 100
    
    print("Impact on ML models:")
    if severe_percentage > 5:
        print("  🔴 CRITICAL: >5% severe outliers will significantly impact most ML algorithms")
        print("     - Tree-based models: Moderate impact")
        print("     - Linear models: High impact")  
        print("     - Neural networks: High impact")
        print("     - Recommendations: Use robust preprocessing or remove severe outliers")
    elif severe_percentage > 2:
        print("  🟡 MODERATE: 2-5% severe outliers require attention")
        print("     - Consider robust scaling or outlier-resistant algorithms")
    else:
        print("  🟢 LOW: <2% severe outliers - manageable with standard techniques")
    
    print(f"\nData Quality Assessment:")
    print(f"  • Overall data quality: {'POOR' if outlier_percentage > 50 else 'MODERATE' if outlier_percentage > 20 else 'GOOD'}")
    print(f"  • Recommended approach: {'Extensive cleaning' if severe_percentage > 10 else 'Targeted cleaning' if severe_percentage > 5 else 'Standard preprocessing'}")
    
    print("\n📊 SUMMARY FOR DAV 6150 MODULE 3:")
    print("-" * 50)
    print("Key Findings:")
    print("1. Dataset shows extensive pre-processing effects")
    print("2. Domain knowledge reveals significant data quality issues")
    print("3. Consensus approach identifies most problematic data points")
    print("4. Treatment recommendations prioritize by severity")
    print("5. Before/after analysis shows treatment effectiveness")
    
    print("\nRecommended Analysis Workflow:")
    print("1. Focus on severe outliers (1,287 points) for immediate review")
    print("2. Apply percentile capping for moderate outliers")
    print("3. Use robust scaling methods for remaining analysis")
    print("4. Validate model performance with and without outlier treatment")
    print("5. Document data quality issues in final analysis report")
    
    print("\n📁 Generated Visualizations:")
    print("   • consensus_outlier_analysis.png - Score distribution and categories")
    print("   • method_comparison_heatmap.png - Detection method comparison")
    print("   • annotated_boxplots.png - Box plots with outlier highlights")
    print("   • multivariate_outlier_scatter.png - 2D multivariate visualization")
    print("   • method_agreement_chart.png - Agreement between methods")
    print("   • before_after_comparison_cap_percentiles.png - Treatment effects")
    
    print("\n" + "="*70)
    print("OUTLIER EXAMINATION COMPLETE")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = examine_outlier_findings()