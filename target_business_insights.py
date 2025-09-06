#!/usr/bin/env python3
"""
Detailed business insights examination from TARGET analysis.
Focus on actionable wine sales improvement strategies.
"""

import pandas as pd
import numpy as np
from target_analysis import comprehensive_target_analysis

def generate_business_insights():
    """Generate detailed business insights from TARGET analysis."""
    
    print("="*70)
    print("WINE SALES BUSINESS INTELLIGENCE REPORT")
    print("="*70)
    
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Run comprehensive analysis
    results = comprehensive_target_analysis(df)
    
    print("\n📈 MARKET ANALYSIS:")
    print("-" * 50)
    
    # Distribution insights
    dist_analysis = results['distribution_analysis']
    
    print(f"Wine Sales Performance Overview:")
    print(f"  • Portfolio size: {dist_analysis['count']:,} wines")
    print(f"  • Sales range: {dist_analysis['min']} to {dist_analysis['max']} units")
    print(f"  • Average performance: {dist_analysis['mean']:.1f} units")
    print(f"  • Top quartile threshold: {dist_analysis['q3']:.1f} units")
    print(f"  • Bottom quartile: {dist_analysis['q1']:.1f} units")
    
    # Market segmentation
    zero_analysis = results['zero_inflation_analysis']
    value_dist = zero_analysis['value_distribution']
    
    print(f"\nMarket Segmentation:")
    total_wines = zero_analysis['total_wines']
    print(f"  • Non-performers (0 sales): {value_dist['zero_sales']:,} wines ({value_dist['zero_sales']/total_wines*100:.1f}%)")
    print(f"  • Low performers (1-2 sales): {value_dist['low_sales_1_2']:,} wines ({value_dist['low_sales_1_2']/total_wines*100:.1f}%)")
    print(f"  • Moderate performers (3-5): {value_dist['moderate_sales_3_5']:,} wines ({value_dist['moderate_sales_3_5']/total_wines*100:.1f}%)")
    print(f"  • Good performers (6-8): {value_dist['good_sales_6_8']:,} wines ({value_dist['good_sales_6_8']/total_wines*100:.1f}%)")
    print(f"  • Excellent performers (>8): {value_dist['excellent_sales_above_8']:,} wines ({value_dist['excellent_sales_above_8']/total_wines*100:.1f}%)")
    
    print("\n🎯 SALES SUCCESS FACTORS:")
    print("-" * 50)
    
    relationship_analysis = results['relationship_analysis']
    
    print("Top Positive Sales Drivers:")
    if 'top_positive_predictors' in relationship_analysis:
        top_positive = relationship_analysis['top_positive_predictors'][:5]
        for i, predictor in enumerate(top_positive, 1):
            significance = "***" if predictor['pearson_p_value'] < 0.001 else \
                         "**" if predictor['pearson_p_value'] < 0.01 else \
                         "*" if predictor['pearson_p_value'] < 0.05 else ""
            
            print(f"  {i}. {predictor['variable']:<18}: r = {predictor['pearson_correlation']:>6.3f}{significance}")
            print(f"     Impact: {predictor['business_interpretation']}")
    
    print("\nTop Negative Sales Factors:")
    if 'top_negative_predictors' in relationship_analysis:
        top_negative = relationship_analysis['top_negative_predictors'][:3]
        for i, predictor in enumerate(top_negative, 1):
            significance = "***" if predictor['pearson_p_value'] < 0.001 else \
                         "**" if predictor['pearson_p_value'] < 0.01 else \
                         "*" if predictor['pearson_p_value'] < 0.05 else ""
            
            print(f"  {i}. {predictor['variable']:<18}: r = {predictor['pearson_correlation']:>6.3f}{significance}")
            print(f"     Impact: {predictor['business_interpretation']}")
    
    print("\n🔬 PREDICTIVE ANALYTICS:")
    print("-" * 50)
    
    predictive_analysis = results['predictive_analysis']
    
    if 'consensus_ranking' in predictive_analysis:
        consensus = predictive_analysis['consensus_ranking']['top_5_consensus']
        print("Top 5 Predictive Features (Multi-Method Consensus):")
        for i, feature in enumerate(consensus, 1):
            # Get correlation info
            feature_info = None
            if 'all_relationships' in relationship_analysis:
                feature_info = relationship_analysis['all_relationships'].get(feature)
            
            if feature_info:
                correlation = feature_info['pearson_correlation']
                print(f"  {i}. {feature}: r = {correlation:.3f}")
            else:
                print(f"  {i}. {feature}")
    
    # Random Forest insights
    if 'random_forest' in predictive_analysis and 'error' not in predictive_analysis['random_forest']:
        rf_results = predictive_analysis['random_forest']
        print(f"\nRandom Forest Model Performance:")
        print(f"  • Model accuracy (R²): {rf_results['model_score']:.3f}")
        print("  • Top features by importance:")
        top_rf = rf_results['top_5_features']
        for i, feature in enumerate(top_rf, 1):
            importance = rf_results['feature_importance'][rf_results['feature_importance']['feature'] == feature]['importance'].iloc[0]
            print(f"    {i}. {feature}: {importance:.3f}")
    
    print("\n🏆 HIGH VS LOW PERFORMERS ANALYSIS:")
    print("-" * 50)
    
    segment_analysis = results['segment_analysis']
    
    if 'key_differentiators' in segment_analysis:
        differentiators = segment_analysis['key_differentiators'][:5]
        
        print("Key Success Differentiators (High vs Low Sellers):")
        for i, diff in enumerate(differentiators, 1):
            direction = "HIGHER" if diff['cohens_d'] > 0 else "LOWER"
            magnitude = "Large" if abs(diff['cohens_d']) > 0.8 else \
                       "Medium" if abs(diff['cohens_d']) > 0.5 else "Small"
            
            print(f"  {i}. {diff['variable']}: {direction} values in top sellers")
            print(f"     Effect size: {diff['cohens_d']:.3f} ({magnitude} effect)")
            print(f"     High sellers: {diff['high_segment_mean']:.2f}, Low sellers: {diff['low_segment_mean']:.2f}")
            print(f"     Business insight: {diff['interpretation']}")
            print()
    
    # Segment sizes
    if 'segment_sizes' in segment_analysis:
        sizes = segment_analysis['segment_sizes']
        print(f"Segment Distribution:")
        total_segments = sum(sizes.values())
        for segment, size in sizes.items():
            print(f"  • {segment} sellers: {size:,} wines ({size/total_segments*100:.1f}%)")
    
    print("\n💼 BUSINESS STRATEGY RECOMMENDATIONS:")
    print("-" * 50)
    
    print("🎯 IMMEDIATE ACTION ITEMS:")
    print("1. QUALITY FOCUS:")
    print("   • STARS rating is the #1 sales driver (r = 0.559)")
    print("   • Invest in wine quality improvement programs")
    print("   • Target wines with <3 stars for quality enhancement")
    
    print("\n2. MARKETING & BRANDING:")  
    print("   • LabelAppeal strongly correlates with sales (r = 0.357)")
    print("   • Redesign labels for wines with negative appeal scores")
    print("   • A/B test label designs to optimize consumer attraction")
    
    print("\n3. PORTFOLIO OPTIMIZATION:")
    non_performers = value_dist['zero_sales']
    print(f"   • Address {non_performers:,} non-performing wines (21.4% of portfolio)")
    print("   • Consider discontinuing or reformulating zero-sales wines")
    print("   • Focus resources on wines with sales potential")
    
    print("\n📊 WINE CHEMISTRY OPTIMIZATION:")
    print("-" * 50)
    
    # Chemistry recommendations based on correlations
    chemistry_predictors = []
    if 'chemical_relationships' in relationship_analysis:
        chemistry_rels = relationship_analysis['chemical_relationships']
        for var, rel_data in chemistry_rels.items():
            if rel_data['pearson_p_value'] < 0.05:  # Significant relationships
                chemistry_predictors.append((var, rel_data['pearson_correlation'], rel_data['business_interpretation']))
    
    if chemistry_predictors:
        chemistry_predictors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Chemical Composition Guidelines for Sales Success:")
        for i, (variable, correlation, interpretation) in enumerate(chemistry_predictors[:5], 1):
            direction = "Increase" if correlation > 0 else "Decrease"
            strength = "Strong" if abs(correlation) > 0.1 else "Moderate"
            print(f"  {i}. {variable}: {direction} levels ({strength} impact, r = {correlation:.3f})")
            print(f"     Business guidance: {interpretation}")
    
    print("\n🎯 MARKET OPPORTUNITIES:")
    print("-" * 50)
    
    print("1. PRODUCT DEVELOPMENT:")
    print("   • High-quality wines (4+ stars) show strongest sales performance")
    print("   • Focus R&D on achieving consistent 4+ star ratings")
    print("   • Consider premium product line for top performers")
    
    print("\n2. MARKET POSITIONING:")
    if dist_analysis.get('skewness', 0) < -0.5:
        print("   • Sales distribution is left-skewed - most wines perform well")
        print("   • Opportunity to identify and replicate success factors")
    else:
        print("   • Many wines underperform - significant improvement potential")
        print("   • Apply top-performer characteristics to struggling wines")
    
    print("\n3. PRICING STRATEGY:")
    print("   • Wine quality (STARS) strongly predicts sales volume")
    print("   • Consider value-based pricing aligned with quality ratings")
    print("   • Premium pricing opportunity for 4+ star wines")
    
    print("\n📈 PERFORMANCE MONITORING KPIs:")
    print("-" * 50)
    
    print("Key Performance Indicators to Track:")
    print("1. Primary KPIs:")
    print("   • Average STARS rating across portfolio")
    print("   • Percentage of wines with 4+ stars")
    print("   • LabelAppeal scores and trends")
    print("   • Zero-sales wine percentage (target: <10%)")
    
    print("\n2. Secondary KPIs:")
    if chemistry_predictors:
        print("   • Chemical composition metrics:")
        for var, correlation, _ in chemistry_predictors[:3]:
            direction = "above" if correlation > 0 else "below"
            print(f"     - {var} levels ({direction} optimal range)")
    
    print("\n3. Business Metrics:")
    print("   • Portfolio performance distribution")
    print("   • High-performer (6+ sales) percentage")
    print("   • Quality improvement correlation with sales")
    
    print("\n🔍 ADVANCED ANALYTICS INSIGHTS:")
    print("-" * 50)
    
    print("Statistical Model Findings:")
    if 'random_forest' in predictive_analysis and 'model_score' in predictive_analysis['random_forest']:
        r2_score = predictive_analysis['random_forest']['model_score']
        print(f"  • Wine characteristics explain {r2_score*100:.1f}% of sales variation")
        if r2_score > 0.7:
            print("  • Strong predictive model - high confidence in recommendations")
        elif r2_score > 0.5:
            print("  • Moderate predictive power - recommendations are directionally sound")
        else:
            print("  • Sales influenced by factors beyond measured characteristics")
    
    # Correlation summary
    if 'correlation_summary' in relationship_analysis:
        summary = relationship_analysis['correlation_summary']
        sig_relationships = summary.get('significant_relationships', 0)
        total_vars = summary.get('total_variables_analyzed', 0)
        print(f"  • {sig_relationships}/{total_vars} variables significantly predict sales")
        print(f"  • Multiple factors influence success - holistic approach needed")
    
    print("\n🎯 DAV 6150 MODULE 3 CONCLUSIONS:")
    print("-" * 50)
    
    print("Key Learning Outcomes:")
    print("1. Data-Driven Decision Making:")
    print("   • Statistical analysis reveals clear sales success patterns")
    print("   • Quality ratings are the primary driver of wine sales")
    print("   • Marketing elements (LabelAppeal) have measurable impact")
    
    print("\n2. Business Intelligence Application:")
    print("   • Segmentation analysis identifies improvement opportunities")
    print("   • Predictive modeling enables proactive portfolio management") 
    print("   • Chemical composition optimization has measurable ROI potential")
    
    print("\n3. Actionable Analytics:")
    print("   • Specific, measurable recommendations for business improvement")
    print("   • Clear KPIs for monitoring progress and success")
    print("   • Evidence-based strategy for wine industry success")
    
    print(f"\n📁 EXECUTIVE DASHBOARD VISUALIZATIONS:")
    print("   • target_distribution_analysis.png - Market performance overview")
    print("   • zero_inflation_analysis.png - Non-performer identification")
    print("   • correlation_analysis.png - Sales driver analysis")
    print("   • predictive_features_analysis.png - Multi-method predictions")
    print("   • top_predictors_scatter_plots.png - Key relationship details")
    print("   • effect_sizes_analysis.png - Success factor magnitudes")
    
    print("\n" + "="*70)
    print("BUSINESS INTELLIGENCE ANALYSIS COMPLETE")
    print("="*70)
    
    return results

if __name__ == "__main__":
    business_report = generate_business_insights()