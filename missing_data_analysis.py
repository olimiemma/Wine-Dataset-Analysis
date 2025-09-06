import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class MissingDataAnalyzer:
    """
    Comprehensive missing data analysis toolkit for wine dataset.
    Provides statistical analysis, pattern detection, and visualizations.
    """
    
    def __init__(self, df, output_dir='missing_data_plots'):
        """
        Initialize the missing data analyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset to analyze
        output_dir : str
            Directory to save visualization plots
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.missing_stats = {}
        self.patterns = {}
        self.placeholder_analysis = {}
        
        # Create output directory if it doesn't exist
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def calculate_missing_statistics(self):
        """
        Calculate comprehensive missing value statistics for each column.
        
        Returns:
        --------
        pandas.DataFrame
            Detailed missing value statistics
        """
        print("Calculating missing value statistics...")
        
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100
        data_types = self.df.dtypes
        unique_values = self.df.nunique()
        
        # Calculate completeness score (percentage of non-missing values)
        completeness = 100 - missing_percentage
        
        # Categorize missing data severity
        def categorize_missing(percentage):
            if percentage == 0:
                return "Complete"
            elif percentage <= 5:
                return "Minimal"
            elif percentage <= 15:
                return "Moderate"
            elif percentage <= 30:
                return "Significant"
            else:
                return "Severe"
        
        missing_severity = missing_percentage.apply(categorize_missing)
        
        stats_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_percentage.round(2),
            'Completeness_Score': completeness.round(2),
            'Data_Type': data_types,
            'Unique_Values': unique_values,
            'Severity_Level': missing_severity
        })
        
        # Sort by missing percentage (descending)
        stats_df = stats_df.sort_values('Missing_Percentage', ascending=False)
        
        self.missing_stats = stats_df
        return stats_df
    
    def identify_missing_patterns(self):
        """
        Identify patterns and correlations in missing data.
        
        Returns:
        --------
        dict
            Dictionary containing pattern analysis results
        """
        print("Identifying missing data patterns...")
        
        # Create missing data indicator matrix
        missing_matrix = self.df.isnull().astype(int)
        
        # Pattern 1: Missing data combinations
        missing_patterns = missing_matrix.value_counts().head(10)
        
        # Pattern 2: Correlation between missing values
        missing_corr = missing_matrix.corr()
        
        # Pattern 3: Columns with similar missing patterns
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.3:  # Threshold for significant correlation
                    high_corr_pairs.append({
                        'Column1': missing_corr.columns[i],
                        'Column2': missing_corr.columns[j],
                        'Correlation': round(corr_val, 3)
                    })
        
        # Pattern 4: Missing data by row analysis
        rows_missing_count = missing_matrix.sum(axis=1)
        rows_missing_stats = {
            'rows_with_no_missing': (rows_missing_count == 0).sum(),
            'rows_with_some_missing': ((rows_missing_count > 0) & (rows_missing_count < len(self.df.columns))).sum(),
            'rows_completely_missing': (rows_missing_count == len(self.df.columns)).sum(),
            'avg_missing_per_row': rows_missing_count.mean(),
            'max_missing_per_row': rows_missing_count.max()
        }
        
        self.patterns = {
            'common_patterns': missing_patterns,
            'missing_correlations': missing_corr,
            'high_correlation_pairs': high_corr_pairs,
            'row_analysis': rows_missing_stats
        }
        
        return self.patterns
    
    def detect_placeholder_values(self):
        """
        Detect potential placeholder values that might represent missing data.
        
        Returns:
        --------
        dict
            Analysis of potential placeholder values
        """
        print("Detecting potential placeholder values...")
        
        placeholder_analysis = {}
        
        for column in self.df.columns:
            col_analysis = {
                'suspected_placeholders': [],
                'zero_values': 0,
                'negative_values': 0,
                'extreme_values': 0
            }
            
            if self.df[column].dtype in ['int64', 'float64']:
                # Check for common numeric placeholders
                zero_count = (self.df[column] == 0).sum()
                negative_count = (self.df[column] < 0).sum()
                
                # Check for extreme values (beyond 3 standard deviations)
                if not self.df[column].isnull().all():
                    mean_val = self.df[column].mean()
                    std_val = self.df[column].std()
                    if std_val > 0:
                        extreme_count = ((self.df[column] - mean_val).abs() > 3 * std_val).sum()
                        col_analysis['extreme_values'] = extreme_count
                
                col_analysis['zero_values'] = zero_count
                col_analysis['negative_values'] = negative_count
                
                # Flag suspicious patterns
                total_values = len(self.df[column]) - self.df[column].isnull().sum()
                if total_values > 0:
                    if zero_count / total_values > 0.1:  # More than 10% zeros
                        col_analysis['suspected_placeholders'].append(f"High zero frequency: {zero_count} ({zero_count/total_values*100:.1f}%)")
                    
                    if negative_count > 0 and column not in ['CitricAcid', 'LabelAppeal']:  # Some columns can naturally be negative
                        col_analysis['suspected_placeholders'].append(f"Unexpected negative values: {negative_count}")
            
            elif self.df[column].dtype == 'object':
                # Check for common string placeholders
                common_placeholders = ['', ' ', 'NULL', 'null', 'None', 'N/A', 'n/a', 'NA', 'missing', 'unknown', '?']
                for placeholder in common_placeholders:
                    count = (self.df[column] == placeholder).sum()
                    if count > 0:
                        col_analysis['suspected_placeholders'].append(f"'{placeholder}': {count} occurrences")
            
            placeholder_analysis[column] = col_analysis
        
        self.placeholder_analysis = placeholder_analysis
        return placeholder_analysis
    
    def create_missing_data_visualizations(self):
        """
        Create comprehensive visualizations for missing data analysis.
        Saves all plots to the specified output directory.
        """
        print("Creating missing data visualizations...")
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Visualization 1: Missing data bar chart
        plt.figure(figsize=(12, 8))
        missing_data = self.df.isnull().sum().sort_values(ascending=True)
        missing_data = missing_data[missing_data > 0]  # Only show columns with missing data
        
        bars = plt.barh(range(len(missing_data)), missing_data.values)
        plt.yticks(range(len(missing_data)), missing_data.index)
        plt.xlabel('Number of Missing Values')
        plt.title('Missing Data Count by Column', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 10, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)} ({width/len(self.df)*100:.1f}%)', 
                    ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/missing_data_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualization 2: Missing data heatmap
        plt.figure(figsize=(14, 10))
        missing_matrix = self.df.isnull()
        
        # Sample data if too large for visualization
        if len(self.df) > 1000:
            sample_indices = np.random.choice(len(self.df), 1000, replace=False)
            sample_indices.sort()
            missing_matrix = missing_matrix.iloc[sample_indices]
        
        sns.heatmap(missing_matrix.T, cbar=True, cmap='viridis', 
                   yticklabels=True, xticklabels=False)
        plt.title('Missing Data Pattern Heatmap\n(Yellow = Missing, Dark = Present)', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Columns')
        plt.xlabel('Data Points (Sample)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/missing_data_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualization 3: Missing data correlation heatmap
        if hasattr(self, 'patterns') and 'missing_correlations' in self.patterns:
            plt.figure(figsize=(12, 10))
            missing_corr = self.patterns['missing_correlations']
            
            # Only show columns with missing data
            cols_with_missing = self.df.isnull().sum()[self.df.isnull().sum() > 0].index
            if len(cols_with_missing) > 1:
                missing_corr_subset = missing_corr.loc[cols_with_missing, cols_with_missing]
                
                mask = np.triu(np.ones_like(missing_corr_subset, dtype=bool))
                sns.heatmap(missing_corr_subset, mask=mask, annot=True, cmap='RdBu_r', 
                           center=0, square=True, fmt='.2f')
                plt.title('Missing Data Correlation Matrix', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/missing_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualization 4: Missing data percentage pie chart
        if hasattr(self, 'missing_stats'):
            plt.figure(figsize=(10, 8))
            severity_counts = self.missing_stats['Severity_Level'].value_counts()
            
            colors = ['#2E8B57', '#FFD700', '#FFA500', '#FF6347', '#DC143C']  # Green to Red spectrum
            plt.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors[:len(severity_counts)])
            plt.title('Distribution of Missing Data Severity Levels', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/missing_severity_pie_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualization 5: Missing data per row distribution
        plt.figure(figsize=(12, 6))
        missing_per_row = self.df.isnull().sum(axis=1)
        plt.hist(missing_per_row, bins=range(missing_per_row.max() + 2), alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Missing Values per Row')
        plt.ylabel('Frequency')
        plt.title('Distribution of Missing Values per Row', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/missing_per_row_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ All visualizations saved to '{self.output_dir}' directory")
    
    def generate_missing_data_report(self):
        """
        Generate a comprehensive missing data analysis report.
        
        Returns:
        --------
        dict
            Complete missing data analysis report
        """
        print("Generating comprehensive missing data report...")
        
        report = {
            'dataset_overview': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'total_cells': self.df.size,
                'total_missing_cells': self.df.isnull().sum().sum(),
                'overall_missing_percentage': round((self.df.isnull().sum().sum() / self.df.size) * 100, 2)
            },
            'column_statistics': self.missing_stats.to_dict('records') if hasattr(self, 'missing_stats') else {},
            'missing_patterns': self.patterns if hasattr(self, 'patterns') else {},
            'placeholder_analysis': self.placeholder_analysis if hasattr(self, 'placeholder_analysis') else {},
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        """
        Generate recommendations based on missing data analysis.
        
        Returns:
        --------
        list
            List of recommendations for handling missing data
        """
        recommendations = []
        
        if hasattr(self, 'missing_stats'):
            # Check overall missing data level
            overall_missing = (self.df.isnull().sum().sum() / self.df.size) * 100
            
            if overall_missing < 5:
                recommendations.append("Overall missing data level is low (<5%). Simple imputation methods should work well.")
            elif overall_missing < 15:
                recommendations.append("Moderate missing data level (5-15%). Consider multiple imputation techniques.")
            else:
                recommendations.append("High missing data level (>15%). Careful analysis needed before imputation.")
            
            # Check for columns with high missing percentages
            high_missing_cols = self.missing_stats[self.missing_stats['Missing_Percentage'] > 30]
            if not high_missing_cols.empty:
                recommendations.append(f"Consider dropping columns with >30% missing data: {list(high_missing_cols['Column'])}")
            
            # Check for complete columns
            complete_cols = self.missing_stats[self.missing_stats['Missing_Percentage'] == 0]
            if not complete_cols.empty:
                recommendations.append(f"Columns with no missing data can be used for imputation: {len(complete_cols)} columns")
        
        if hasattr(self, 'patterns') and self.patterns['high_correlation_pairs']:
            recommendations.append("High correlation found between missing values in some columns. Consider multivariate imputation.")
        
        # Check for placeholder values
        if hasattr(self, 'placeholder_analysis'):
            suspicious_cols = [col for col, analysis in self.placeholder_analysis.items() 
                             if analysis['suspected_placeholders']]
            if suspicious_cols:
                recommendations.append(f"Investigate potential placeholder values in: {suspicious_cols}")
        
        return recommendations

def analyze_missing_data(df, output_dir='missing_data_plots'):
    """
    Comprehensive missing data analysis function.
    
    This function performs a complete missing data analysis including:
    - Statistical analysis of missing values
    - Pattern identification and correlations
    - Placeholder value detection
    - Comprehensive visualizations
    - Detailed reporting with recommendations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    output_dir : str, default 'missing_data_plots'
        Directory to save visualization plots
    
    Returns:
    --------
    dict
        Comprehensive missing data analysis report
    """
    print("="*70)
    print("COMPREHENSIVE MISSING DATA ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = MissingDataAnalyzer(df, output_dir)
    
    # Run all analyses
    analyzer.calculate_missing_statistics()
    analyzer.identify_missing_patterns()
    analyzer.detect_placeholder_values()
    analyzer.create_missing_data_visualizations()
    
    # Generate final report
    report = analyzer.generate_missing_data_report()
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   • Total missing cells: {report['dataset_overview']['total_missing_cells']:,}")
    print(f"   • Overall missing percentage: {report['dataset_overview']['overall_missing_percentage']}%")
    print(f"   • Columns with missing data: {len([col for col in analyzer.missing_stats['Column'] if analyzer.missing_stats[analyzer.missing_stats['Column'] == col]['Missing_Count'].iloc[0] > 0])}")
    print(f"   • Visualizations saved to: {output_dir}")
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "="*70)
    print("MISSING DATA ANALYSIS COMPLETE")
    print("="*70)
    
    return report

# Example usage
if __name__ == "__main__":
    # Load the wine dataset for testing
    from wine_data_assessment import load_and_assess_data
    
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Perform missing data analysis
    report = analyze_missing_data(df)