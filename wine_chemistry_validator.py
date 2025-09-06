import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class WineChemistryValidator:
    """
    Wine chemistry domain validation toolkit based on established enological science.
    
    This validator uses scientifically established ranges for wine chemical parameters
    to identify data quality issues, impossible values, and suspicious outliers.
    """
    
    def __init__(self):
        """Initialize wine chemistry parameter ranges based on enological literature."""
        
        # Wine chemistry parameter ranges based on scientific literature
        # Sources: OIV standards, Davis Wine Science, Jackson "Wine Science"
        self.parameter_ranges = {
            'pH': {
                'normal_range': (2.9, 4.0),
                'extreme_range': (2.5, 4.5),
                'typical_range': (3.0, 3.8),
                'description': 'Wine pH - critical for stability, taste, and microbiology',
                'units': 'pH units',
                'white_wine_typical': (3.0, 3.4),
                'red_wine_typical': (3.3, 3.8)
            },
            
            'Alcohol': {
                'normal_range': (8.0, 16.0),
                'extreme_range': (5.0, 20.0),
                'typical_range': (11.0, 14.5),
                'description': 'Alcohol by volume - legal and practical limits',
                'units': '% vol',
                'table_wine_range': (8.5, 15.0),
                'fortified_max': 20.0
            },
            
            'FixedAcidity': {
                'normal_range': (4.0, 12.0),
                'extreme_range': (2.0, 18.0),
                'typical_range': (6.0, 9.0),
                'description': 'Fixed acidity - primarily tartaric acid',
                'units': 'g/L as tartaric acid',
                'white_wine_typical': (5.5, 8.5),
                'red_wine_typical': (6.0, 9.5)
            },
            
            'VolatileAcidity': {
                'normal_range': (0.1, 1.2),
                'extreme_range': (0.0, 2.0),
                'typical_range': (0.3, 0.8),
                'description': 'Volatile acidity - primarily acetic acid',
                'units': 'g/L as acetic acid',
                'legal_limit_table': 1.2,
                'spoilage_threshold': 1.0
            },
            
            'CitricAcid': {
                'normal_range': (0.0, 1.0),
                'extreme_range': (0.0, 2.0),
                'typical_range': (0.1, 0.6),
                'description': 'Citric acid content - naturally occurring or added',
                'units': 'g/L',
                'naturally_occurring_max': 0.5
            },
            
            'ResidualSugar': {
                'normal_range': (0.5, 150.0),
                'extreme_range': (0.0, 300.0),
                'typical_range': (1.0, 10.0),
                'description': 'Residual sugar content',
                'units': 'g/L',
                'dry_wine_max': 4.0,
                'off_dry_range': (4.0, 12.0),
                'sweet_wine_min': 45.0
            },
            
            'Chlorides': {
                'normal_range': (0.01, 0.8),
                'extreme_range': (0.0, 2.0),
                'typical_range': (0.05, 0.3),
                'description': 'Chloride content - indicator of terroir and processing',
                'units': 'g/L as NaCl',
                'coastal_regions_higher': True
            },
            
            'FreeSulfurDioxide': {
                'normal_range': (5.0, 80.0),
                'extreme_range': (0.0, 100.0),
                'typical_range': (20.0, 50.0),
                'description': 'Free SO2 - antimicrobial and antioxidant',
                'units': 'mg/L',
                'white_wine_target': (25.0, 40.0),
                'red_wine_target': (15.0, 25.0)
            },
            
            'TotalSulfurDioxide': {
                'normal_range': (20.0, 350.0),
                'extreme_range': (10.0, 500.0),
                'typical_range': (80.0, 200.0),
                'description': 'Total SO2 - includes free and bound forms',
                'units': 'mg/L',
                'legal_limit_red': 160.0,
                'legal_limit_white': 210.0,
                'organic_limit': 100.0
            },
            
            'Density': {
                'normal_range': (0.985, 1.005),
                'extreme_range': (0.980, 1.020),
                'typical_range': (0.990, 1.000),
                'description': 'Wine density - related to sugar and alcohol content',
                'units': 'g/mL',
                'dry_wine_range': (0.985, 0.998),
                'sweet_wine_range': (0.995, 1.005)
            },
            
            'Sulphates': {
                'normal_range': (0.3, 2.0),
                'extreme_range': (0.1, 3.0),
                'typical_range': (0.4, 1.2),
                'description': 'Sulfate content - affects SO2 effectiveness',
                'units': 'g/L as K2SO4',
                'legal_limit': 2.0
            }
        }
        
        # Quality flags for different violation types
        self.quality_flags = {
            'IMPOSSIBLE': 'Chemically/physically impossible value',
            'EXTREME_OUTLIER': 'Value outside extreme range for wine',
            'SUSPICIOUS': 'Unusual but possible value requiring verification',
            'LOGICAL_VIOLATION': 'Violates logical relationships between parameters',
            'REGULATORY_VIOLATION': 'Exceeds legal/regulatory limits',
            'STYLE_INCONSISTENT': 'Inconsistent with typical wine styles'
        }
    
    def validate_parameter_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate wine chemistry parameters against established ranges.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Wine dataset with chemistry parameters
            
        Returns:
        --------
        dict
            Validation results with flags and statistics
        """
        print("Validating wine chemistry parameter ranges...")
        
        validation_results = {
            'parameter_violations': {},
            'summary_stats': {},
            'flagged_rows': set(),
            'total_violations': 0
        }
        
        for param, ranges in self.parameter_ranges.items():
            if param not in df.columns:
                continue
                
            param_data = df[param].dropna()
            if len(param_data) == 0:
                continue
            
            violations = {
                'impossible_values': [],
                'extreme_outliers': [],
                'suspicious_values': [],
                'count_impossible': 0,
                'count_extreme': 0,
                'count_suspicious': 0
            }
            
            # Check for impossible values (outside extreme range)
            extreme_min, extreme_max = ranges['extreme_range']
            impossible_mask = (param_data < extreme_min) | (param_data > extreme_max)
            violations['impossible_values'] = param_data.index[impossible_mask].tolist()
            violations['count_impossible'] = impossible_mask.sum()
            
            # Check for extreme outliers (outside normal range but within extreme)
            normal_min, normal_max = ranges['normal_range']
            extreme_mask = ((param_data < normal_min) & (param_data >= extreme_min)) | \
                          ((param_data > normal_max) & (param_data <= extreme_max))
            violations['extreme_outliers'] = param_data.index[extreme_mask].tolist()
            violations['count_extreme'] = extreme_mask.sum()
            
            # Check for suspicious values (outside typical range but within normal)
            typical_min, typical_max = ranges['typical_range']
            suspicious_mask = ((param_data < typical_min) & (param_data >= normal_min)) | \
                             ((param_data > typical_max) & (param_data <= normal_max))
            violations['suspicious_values'] = param_data.index[suspicious_mask].tolist()
            violations['count_suspicious'] = suspicious_mask.sum()
            
            # Update flagged rows
            validation_results['flagged_rows'].update(violations['impossible_values'])
            validation_results['flagged_rows'].update(violations['extreme_outliers'])
            
            validation_results['parameter_violations'][param] = violations
            validation_results['total_violations'] += violations['count_impossible'] + violations['count_extreme']
            
            # Summary statistics
            validation_results['summary_stats'][param] = {
                'min': param_data.min(),
                'max': param_data.max(),
                'mean': param_data.mean(),
                'std': param_data.std(),
                'count_total': len(param_data),
                'percent_impossible': (violations['count_impossible'] / len(param_data)) * 100,
                'percent_extreme': (violations['count_extreme'] / len(param_data)) * 100,
                'percent_suspicious': (violations['count_suspicious'] / len(param_data)) * 100
            }
        
        return validation_results
    
    def check_logical_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check logical relationships between wine chemistry parameters.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Wine dataset with chemistry parameters
            
        Returns:
        --------
        dict
            Logical relationship violation results
        """
        print("Checking logical relationships between parameters...")
        
        logical_violations = {
            'total_so2_vs_free_so2': [],
            'density_vs_alcohol': [],
            'acidity_relationships': [],
            'sugar_vs_density': [],
            'summary': {}
        }
        
        # Rule 1: Total SO2 must be >= Free SO2
        if 'TotalSulfurDioxide' in df.columns and 'FreeSulfurDioxide' in df.columns:
            mask = (df['TotalSulfurDioxide'] < df['FreeSulfurDioxide']) & \
                   df['TotalSulfurDioxide'].notna() & df['FreeSulfurDioxide'].notna()
            logical_violations['total_so2_vs_free_so2'] = df.index[mask].tolist()
        
        # Rule 2: Higher alcohol typically means lower density (inverse relationship)
        if 'Alcohol' in df.columns and 'Density' in df.columns:
            # Flag cases where high alcohol (>14%) has high density (>0.998)
            mask = (df['Alcohol'] > 14.0) & (df['Density'] > 0.998) & \
                   df['Alcohol'].notna() & df['Density'].notna()
            logical_violations['density_vs_alcohol'] = df.index[mask].tolist()
        
        # Rule 3: Very low fixed acidity with very high pH is suspicious
        if 'FixedAcidity' in df.columns and 'pH' in df.columns:
            mask = (df['FixedAcidity'] < 4.0) & (df['pH'] > 3.8) & \
                   df['FixedAcidity'].notna() & df['pH'].notna()
            logical_violations['acidity_relationships'] = df.index[mask].tolist()
        
        # Rule 4: High residual sugar should correlate with higher density
        if 'ResidualSugar' in df.columns and 'Density' in df.columns:
            # Flag high sugar (>20g/L) with very low density (<0.992)
            mask = (df['ResidualSugar'] > 20.0) & (df['Density'] < 0.992) & \
                   df['ResidualSugar'].notna() & df['Density'].notna()
            logical_violations['sugar_vs_density'] = df.index[mask].tolist()
        
        # Summary statistics
        logical_violations['summary'] = {
            'total_so2_violations': len(logical_violations['total_so2_vs_free_so2']),
            'density_alcohol_violations': len(logical_violations['density_vs_alcohol']),
            'acidity_violations': len(logical_violations['acidity_relationships']),
            'sugar_density_violations': len(logical_violations['sugar_vs_density']),
            'total_logical_violations': sum([
                len(logical_violations['total_so2_vs_free_so2']),
                len(logical_violations['density_vs_alcohol']),
                len(logical_violations['acidity_relationships']),
                len(logical_violations['sugar_vs_density'])
            ])
        }
        
        return logical_violations
    
    def detect_wine_style_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect values inconsistent with typical wine styles and production methods.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Wine dataset with chemistry parameters
            
        Returns:
        --------
        dict
            Wine style inconsistency results
        """
        print("Detecting wine style inconsistencies...")
        
        style_issues = {
            'impossible_combinations': [],
            'regulatory_violations': [],
            'production_anomalies': [],
            'summary': {}
        }
        
        # Check for impossible combinations
        impossible_conditions = []
        
        # Very high alcohol with very high residual sugar (fortified wine indicators)
        if 'Alcohol' in df.columns and 'ResidualSugar' in df.columns:
            mask = (df['Alcohol'] > 15.0) & (df['ResidualSugar'] > 100.0) & \
                   df['Alcohol'].notna() & df['ResidualSugar'].notna()
            impossible_conditions.extend(df.index[mask].tolist())
        
        # Check regulatory violations
        regulatory_violations = []
        
        # Volatile acidity exceeding legal limits
        if 'VolatileAcidity' in df.columns:
            mask = df['VolatileAcidity'] > 1.2  # Legal limit for table wines
            regulatory_violations.extend(df.index[mask & df['VolatileAcidity'].notna()].tolist())
        
        # Total SO2 exceeding typical legal limits
        if 'TotalSulfurDioxide' in df.columns:
            mask = df['TotalSulfurDioxide'] > 350.0  # Conservative legal limit
            regulatory_violations.extend(df.index[mask & df['TotalSulfurDioxide'].notna()].tolist())
        
        # Production anomalies
        production_anomalies = []
        
        # Citric acid levels suggesting artificial addition beyond normal limits
        if 'CitricAcid' in df.columns:
            mask = df['CitricAcid'] > 1.0  # Above typical addition levels
            production_anomalies.extend(df.index[mask & df['CitricAcid'].notna()].tolist())
        
        style_issues['impossible_combinations'] = list(set(impossible_conditions))
        style_issues['regulatory_violations'] = list(set(regulatory_violations))
        style_issues['production_anomalies'] = list(set(production_anomalies))
        
        style_issues['summary'] = {
            'impossible_combinations_count': len(style_issues['impossible_combinations']),
            'regulatory_violations_count': len(style_issues['regulatory_violations']),
            'production_anomalies_count': len(style_issues['production_anomalies']),
            'total_style_issues': len(set(impossible_conditions + regulatory_violations + production_anomalies))
        }
        
        return style_issues
    
    def create_quality_flags(self, df: pd.DataFrame, validation_results: Dict, 
                           logical_results: Dict, style_results: Dict) -> pd.DataFrame:
        """
        Create comprehensive quality flags for the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Original wine dataset
        validation_results : dict
            Parameter range validation results
        logical_results : dict
            Logical relationship validation results
        style_results : dict
            Wine style consistency results
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with added quality flag columns
        """
        print("Creating comprehensive quality flags...")
        
        df_flagged = df.copy()
        
        # Initialize flag columns
        df_flagged['QF_IMPOSSIBLE'] = False
        df_flagged['QF_EXTREME_OUTLIER'] = False
        df_flagged['QF_SUSPICIOUS'] = False
        df_flagged['QF_LOGICAL_VIOLATION'] = False
        df_flagged['QF_REGULATORY_VIOLATION'] = False
        df_flagged['QF_STYLE_INCONSISTENT'] = False
        df_flagged['QF_OVERALL_QUALITY'] = 'GOOD'
        df_flagged['QF_VIOLATION_COUNT'] = 0
        df_flagged['QF_VIOLATION_DETAILS'] = ''
        
        # Set impossible value flags
        for param, violations in validation_results['parameter_violations'].items():
            df_flagged.loc[violations['impossible_values'], 'QF_IMPOSSIBLE'] = True
            df_flagged.loc[violations['extreme_outliers'], 'QF_EXTREME_OUTLIER'] = True
            df_flagged.loc[violations['suspicious_values'], 'QF_SUSPICIOUS'] = True
        
        # Set logical violation flags
        all_logical_violations = []
        for violation_type, indices in logical_results.items():
            if isinstance(indices, list) and violation_type != 'summary':
                all_logical_violations.extend(indices)
        
        df_flagged.loc[all_logical_violations, 'QF_LOGICAL_VIOLATION'] = True
        
        # Set style inconsistency flags
        df_flagged.loc[style_results['regulatory_violations'], 'QF_REGULATORY_VIOLATION'] = True
        
        all_style_issues = (style_results['impossible_combinations'] + 
                           style_results['production_anomalies'])
        df_flagged.loc[all_style_issues, 'QF_STYLE_INCONSISTENT'] = True
        
        # Calculate violation counts and overall quality
        flag_columns = ['QF_IMPOSSIBLE', 'QF_EXTREME_OUTLIER', 'QF_LOGICAL_VIOLATION', 
                       'QF_REGULATORY_VIOLATION', 'QF_STYLE_INCONSISTENT']
        
        df_flagged['QF_VIOLATION_COUNT'] = df_flagged[flag_columns].sum(axis=1)
        
        # Set overall quality assessment
        df_flagged.loc[df_flagged['QF_IMPOSSIBLE'], 'QF_OVERALL_QUALITY'] = 'CRITICAL'
        df_flagged.loc[(df_flagged['QF_EXTREME_OUTLIER']) | 
                      (df_flagged['QF_LOGICAL_VIOLATION']) |
                      (df_flagged['QF_REGULATORY_VIOLATION']), 'QF_OVERALL_QUALITY'] = 'POOR'
        df_flagged.loc[(df_flagged['QF_SUSPICIOUS']) | 
                      (df_flagged['QF_STYLE_INCONSISTENT']), 'QF_OVERALL_QUALITY'] = 'QUESTIONABLE'
        
        return df_flagged
    
    def generate_validation_report(self, validation_results: Dict, logical_results: Dict, 
                                 style_results: Dict, df_flagged: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        validation_results : dict
            Parameter validation results
        logical_results : dict
            Logical relationship results
        style_results : dict
            Style consistency results
        df_flagged : pandas.DataFrame
            Dataset with quality flags
            
        Returns:
        --------
        dict
            Comprehensive validation report
        """
        total_rows = len(df_flagged)
        
        quality_summary = df_flagged['QF_OVERALL_QUALITY'].value_counts()
        
        report = {
            'dataset_summary': {
                'total_rows': total_rows,
                'rows_with_issues': (df_flagged['QF_VIOLATION_COUNT'] > 0).sum(),
                'percentage_with_issues': ((df_flagged['QF_VIOLATION_COUNT'] > 0).sum() / total_rows) * 100
            },
            'quality_distribution': quality_summary.to_dict(),
            'violation_summary': {
                'impossible_values': (df_flagged['QF_IMPOSSIBLE']).sum(),
                'extreme_outliers': (df_flagged['QF_EXTREME_OUTLIER']).sum(),
                'logical_violations': (df_flagged['QF_LOGICAL_VIOLATION']).sum(),
                'regulatory_violations': (df_flagged['QF_REGULATORY_VIOLATION']).sum(),
                'style_inconsistencies': (df_flagged['QF_STYLE_INCONSISTENT']).sum()
            },
            'parameter_analysis': validation_results['summary_stats'],
            'logical_relationships': logical_results['summary'],
            'style_analysis': style_results['summary'],
            'recommendations': self._generate_validation_recommendations(
                validation_results, logical_results, style_results, df_flagged)
        }
        
        return report

    def _generate_validation_recommendations(self, validation_results: Dict, 
                                           logical_results: Dict, style_results: Dict,
                                           df_flagged: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        
        recommendations = []
        
        # Check for critical issues
        critical_count = (df_flagged['QF_OVERALL_QUALITY'] == 'CRITICAL').sum()
        if critical_count > 0:
            recommendations.append(f"CRITICAL: {critical_count} rows have impossible values - investigate data source and collection methods")
        
        # Check parameter-specific issues
        for param, violations in validation_results['parameter_violations'].items():
            if violations['count_impossible'] > 0:
                recommendations.append(f"Parameter {param}: {violations['count_impossible']} impossible values detected - check measurement accuracy")
        
        # Check logical relationship issues
        if logical_results['summary']['total_logical_violations'] > 0:
            recommendations.append("Logical inconsistencies detected - verify measurement procedures and data entry")
        
        # Check regulatory issues
        if style_results['summary']['regulatory_violations_count'] > 0:
            recommendations.append("Regulatory violations detected - some wines may not meet legal standards")
        
        # Overall data quality assessment
        issue_percentage = ((df_flagged['QF_VIOLATION_COUNT'] > 0).sum() / len(df_flagged)) * 100
        if issue_percentage > 20:
            recommendations.append("High percentage of data quality issues (>20%) - comprehensive data audit recommended")
        elif issue_percentage > 10:
            recommendations.append("Moderate data quality issues (10-20%) - targeted validation of flagged records needed")
        else:
            recommendations.append("Good overall data quality (<10% issues) - routine monitoring sufficient")
        
        return recommendations

def validate_wine_chemistry(df: pd.DataFrame, output_dir: str = 'wine_validation_plots') -> Dict[str, Any]:
    """
    Comprehensive wine chemistry domain validation function.
    
    This function performs extensive validation of wine chemistry data against
    established enological science parameters, detecting impossible values,
    logical inconsistencies, and regulatory violations.
    
    Wine Chemistry Parameter Ranges (based on scientific literature):
    
    pH: 2.9-4.0 (normal), 3.0-3.8 (typical)
    - Critical for wine stability, microbial safety, and taste
    - White wines: 3.0-3.4, Red wines: 3.3-3.8
    
    Alcohol: 8.0-16.0% vol (normal), 11.0-14.5% (typical)  
    - Legal limits vary by region and wine type
    - Table wines: 8.5-15.0%, Fortified: up to 20%
    
    Fixed Acidity: 4.0-12.0 g/L (normal), 6.0-9.0 g/L (typical)
    - Primarily tartaric acid, affects taste and stability
    
    Volatile Acidity: 0.1-1.2 g/L (normal), 0.3-0.8 g/L (typical)
    - Primarily acetic acid, legal limit 1.2 g/L for table wines
    - Above 1.0 g/L indicates spoilage
    
    Residual Sugar: 0.5-150.0 g/L (normal)
    - Dry wines: <4 g/L, Off-dry: 4-12 g/L, Sweet: >45 g/L
    
    Total/Free SO2: Complex relationship, Total ≥ Free always
    - Legal limits: Red wines 160 mg/L, White wines 210 mg/L
    
    Density: 0.985-1.005 g/mL (normal)
    - Inversely related to alcohol, directly to sugar content
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wine dataset with chemistry parameters
    output_dir : str, default 'wine_validation_plots'
        Directory to save validation visualizations
        
    Returns:
    --------
    dict
        Comprehensive validation report including:
        - Parameter range violations
        - Logical relationship violations  
        - Wine style inconsistencies
        - Quality flags and recommendations
        - Flagged dataset with quality indicators
    """
    
    print("="*70)
    print("COMPREHENSIVE WINE CHEMISTRY DOMAIN VALIDATION")
    print("="*70)
    
    # Initialize validator
    validator = WineChemistryValidator()
    
    # Run all validation checks
    print("\n🔍 PARAMETER RANGE VALIDATION")
    print("-" * 50)
    validation_results = validator.validate_parameter_ranges(df)
    
    print("\n🔗 LOGICAL RELATIONSHIP VALIDATION") 
    print("-" * 50)
    logical_results = validator.check_logical_relationships(df)
    
    print("\n🍷 WINE STYLE CONSISTENCY VALIDATION")
    print("-" * 50)
    style_results = validator.detect_wine_style_inconsistencies(df)
    
    # Create quality flags
    print("\n🏷️  QUALITY FLAG GENERATION")
    print("-" * 50)
    df_flagged = validator.create_quality_flags(df, validation_results, logical_results, style_results)
    
    # Generate comprehensive report
    print("\n📋 GENERATING VALIDATION REPORT")
    print("-" * 50)
    report = validator.generate_validation_report(validation_results, logical_results, 
                                                style_results, df_flagged)
    
    # Add flagged dataset to report
    report['flagged_dataset'] = df_flagged
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   • Total rows analyzed: {report['dataset_summary']['total_rows']:,}")
    print(f"   • Rows with quality issues: {report['dataset_summary']['rows_with_issues']:,} ({report['dataset_summary']['percentage_with_issues']:.1f}%)")
    print(f"   • Critical quality issues: {report['violation_summary']['impossible_values']:,}")
    print(f"   • Logical violations: {report['violation_summary']['logical_violations']:,}")
    print(f"   • Regulatory violations: {report['violation_summary']['regulatory_violations']:,}")
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "="*70)
    print("WINE CHEMISTRY VALIDATION COMPLETE")
    print("="*70)
    
    return report

# Example usage
if __name__ == "__main__":
    # Load wine dataset for testing
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Perform wine chemistry validation
    validation_report = validate_wine_chemistry(df)