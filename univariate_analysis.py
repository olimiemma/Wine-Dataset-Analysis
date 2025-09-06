import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, anderson, normaltest, jarque_bera, kstest
import warnings
import os
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

class UnivariateAnalyzer:
    """
    Comprehensive univariate analysis toolkit for numerical data.
    
    Provides extensive descriptive statistics, normality testing,
    professional visualizations, and distribution analysis.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'univariate_analysis_plots'):
        """
        Initialize the univariate analyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to analyze
        output_dir : str
            Directory to save visualization plots
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.numerical_columns = self._identify_numerical_columns()
        self.results = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _identify_numerical_columns(self) -> List[str]:
        """Identify numerical columns suitable for univariate analysis."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove index-like columns
        numerical_cols = [col for col in numerical_cols if not col.upper().startswith('INDEX')]
        return numerical_cols
    
    def calculate_descriptive_statistics(self, column: str) -> Dict[str, float]:
        """
        Calculate comprehensive descriptive statistics for a column.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
            
        Returns:
        --------
        dict
            Dictionary with extensive descriptive statistics
        """
        data = self.df[column].dropna()
        
        if len(data) == 0:
            return {'error': 'No valid data points'}
        
        # Basic statistics
        stats_dict = {
            'count': len(data),
            'missing_count': self.df[column].isnull().sum(),
            'missing_percentage': (self.df[column].isnull().sum() / len(self.df)) * 100,
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
            'std': data.std(),
            'variance': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min()
        }
        
        # Percentiles and quartiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats_dict[f'percentile_{p}'] = data.quantile(p/100)
        
        # IQR and outlier analysis
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        stats_dict['iqr'] = iqr
        
        # Outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        stats_dict['outlier_count'] = len(outliers)
        stats_dict['outlier_percentage'] = (len(outliers) / len(data)) * 100
        stats_dict['outlier_lower_bound'] = lower_bound
        stats_dict['outlier_upper_bound'] = upper_bound
        
        # Shape statistics
        stats_dict['skewness'] = data.skew()
        stats_dict['kurtosis'] = data.kurtosis()
        stats_dict['excess_kurtosis'] = data.kurtosis() - 3  # Excess kurtosis (normal = 0)
        
        # Coefficient of variation
        if stats_dict['mean'] != 0:
            stats_dict['cv'] = (stats_dict['std'] / abs(stats_dict['mean'])) * 100
        else:
            stats_dict['cv'] = np.inf
        
        # Distribution shape interpretation
        stats_dict['skewness_interpretation'] = self._interpret_skewness(stats_dict['skewness'])
        stats_dict['kurtosis_interpretation'] = self._interpret_kurtosis(stats_dict['excess_kurtosis'])
        
        return stats_dict
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness values."""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif -1 < skewness < -0.5:
            return "Moderately left-skewed"
        elif 0.5 < skewness < 1:
            return "Moderately right-skewed"
        elif skewness <= -1:
            return "Highly left-skewed"
        elif skewness >= 1:
            return "Highly right-skewed"
        else:
            return "Symmetric"
    
    def _interpret_kurtosis(self, excess_kurtosis: float) -> str:
        """Interpret excess kurtosis values."""
        if abs(excess_kurtosis) < 0.5:
            return "Mesokurtic (normal-like)"
        elif excess_kurtosis > 0.5:
            return "Leptokurtic (heavy-tailed)"
        else:
            return "Platykurtic (light-tailed)"
    
    def test_normality(self, column: str) -> Dict[str, Any]:
        """
        Perform multiple normality tests on a column.
        
        Parameters:
        -----------
        column : str
            Column name to test
            
        Returns:
        --------
        dict
            Results from multiple normality tests
        """
        data = self.df[column].dropna()
        
        if len(data) < 3:
            return {'error': 'Insufficient data for normality testing'}
        
        normality_results = {
            'sample_size': len(data),
            'tests': {}
        }
        
        # Shapiro-Wilk Test (most powerful for small samples)
        if len(data) <= 5000:  # Shapiro-Wilk has sample size limitations
            try:
                stat, p_value = shapiro(data)
                normality_results['tests']['shapiro_wilk'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05,
                    'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal',
                    'note': 'Most powerful test for small samples (n≤5000)'
                }
            except Exception as e:
                normality_results['tests']['shapiro_wilk'] = {'error': str(e)}
        
        # Anderson-Darling Test
        try:
            result = anderson(data, dist='norm')
            # Use 5% significance level (index 2 corresponds to 5%)
            critical_value = result.critical_values[2]
            is_normal = result.statistic < critical_value
            
            normality_results['tests']['anderson_darling'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_levels.tolist(),
                'is_normal_5pct': is_normal,
                'interpretation': 'Normal' if is_normal else 'Not Normal',
                'note': 'Good for detecting departures in tails'
            }
        except Exception as e:
            normality_results['tests']['anderson_darling'] = {'error': str(e)}
        
        # D'Agostino-Pearson Test
        try:
            stat, p_value = normaltest(data)
            normality_results['tests']['dagostino_pearson'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal',
                'note': 'Tests skewness and kurtosis'
            }
        except Exception as e:
            normality_results['tests']['dagostino_pearson'] = {'error': str(e)}
        
        # Jarque-Bera Test
        try:
            stat, p_value = jarque_bera(data)
            normality_results['tests']['jarque_bera'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal',
                'note': 'Based on skewness and kurtosis, good for large samples'
            }
        except Exception as e:
            normality_results['tests']['jarque_bera'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov Test (against normal distribution)
        try:
            # Standardize data for comparison with standard normal
            standardized_data = (data - data.mean()) / data.std()
            stat, p_value = kstest(standardized_data, 'norm')
            normality_results['tests']['kolmogorov_smirnov'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal',
                'note': 'Tests overall distribution shape'
            }
        except Exception as e:
            normality_results['tests']['kolmogorov_smirnov'] = {'error': str(e)}
        
        # Overall normality assessment
        normal_test_count = 0
        total_test_count = 0
        
        for test_name, test_result in normality_results['tests'].items():
            if 'error' not in test_result:
                total_test_count += 1
                if test_result.get('is_normal', False) or test_result.get('is_normal_5pct', False):
                    normal_test_count += 1
        
        if total_test_count > 0:
            normality_percentage = (normal_test_count / total_test_count) * 100
            normality_results['overall_assessment'] = {
                'tests_passed': normal_test_count,
                'total_tests': total_test_count,
                'percentage_normal': normality_percentage,
                'conclusion': self._interpret_normality(normality_percentage)
            }
        
        return normality_results
    
    def _interpret_normality(self, percentage: float) -> str:
        """Interpret overall normality based on percentage of tests passed."""
        if percentage >= 80:
            return "Likely normal distribution"
        elif percentage >= 50:
            return "Possibly normal, mixed results"
        elif percentage >= 20:
            return "Likely not normal, some conflicting evidence"
        else:
            return "Clearly not normal distribution"
    
    def create_distribution_plots(self, column: str) -> None:
        """
        Create comprehensive distribution plots for a column.
        
        Parameters:
        -----------
        column : str
            Column name to plot
        """
        data = self.df[column].dropna()
        
        if len(data) == 0:
            print(f"No valid data for {column}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Distribution Analysis: {column}', fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Histogram with KDE
        axes[0, 0].hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        # Add KDE if data is suitable
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            axes[0, 0].plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
            axes[0, 0].legend()
        except:
            pass
        
        axes[0, 0].set_title('Histogram with Kernel Density Estimate')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Box Plot
        box_plot = axes[0, 1].boxplot(data, patch_artist=True, notch=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        axes[0, 1].set_title('Box Plot with Outlier Detection')
        axes[0, 1].set_ylabel(column)
        axes[0, 1].grid(alpha=0.3)
        
        # Add statistics annotations to box plot
        stats_text = f'Median: {data.median():.3f}\nIQR: {data.quantile(0.75) - data.quantile(0.25):.3f}'
        axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Plot 3: Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Violin Plot
        axes[1, 1].violinplot(data, positions=[0], showmeans=True, showmedians=True)
        axes[1, 1].set_title('Violin Plot (Distribution Shape)')
        axes[1, 1].set_ylabel(column)
        axes[1, 1].set_xticks([0])
        axes[1, 1].set_xticklabels([column])
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot
        filename = f'{self.output_dir}/distribution_{column.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional detailed histogram
        self._create_detailed_histogram(column, data)
    
    def _create_detailed_histogram(self, column: str, data: pd.Series) -> None:
        """Create a detailed histogram with statistical annotations."""
        
        plt.figure(figsize=(12, 8))
        
        # Calculate optimal number of bins
        n_bins = min(50, max(10, int(np.sqrt(len(data)))))
        
        # Create histogram
        n, bins, patches = plt.hist(data, bins=n_bins, density=True, alpha=0.7, 
                                   color='lightblue', edgecolor='black', linewidth=0.5)
        
        # Add KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            plt.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
        except:
            pass
        
        # Add normal distribution overlay for comparison
        mu, sigma = data.mean(), data.std()
        x_norm = np.linspace(data.min(), data.max(), 200)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        plt.plot(x_norm, y_norm, color='green', linewidth=2, linestyle='--', 
                label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        # Add vertical lines for key statistics
        plt.axvline(data.mean(), color='red', linestyle='-', alpha=0.8, label=f'Mean: {data.mean():.3f}')
        plt.axvline(data.median(), color='orange', linestyle='-', alpha=0.8, label=f'Median: {data.median():.3f}')
        
        # Add quartile lines
        plt.axvline(data.quantile(0.25), color='gray', linestyle=':', alpha=0.6, label='Q1')
        plt.axvline(data.quantile(0.75), color='gray', linestyle=':', alpha=0.6, label='Q3')
        
        plt.title(f'Detailed Distribution: {column}', fontsize=14, fontweight='bold')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add statistical summary box
        stats_text = (f'Count: {len(data):,}\n'
                     f'Mean: {data.mean():.3f}\n'
                     f'Std: {data.std():.3f}\n'
                     f'Skewness: {data.skew():.3f}\n'
                     f'Kurtosis: {data.kurtosis():.3f}')
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save detailed histogram
        filename = f'{self.output_dir}/detailed_histogram_{column.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_single_variable(self, column: str) -> Dict[str, Any]:
        """
        Perform complete univariate analysis for a single variable.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        print(f"Analyzing {column}...")
        
        analysis_results = {
            'column': column,
            'descriptive_statistics': self.calculate_descriptive_statistics(column),
            'normality_tests': self.test_normality(column)
        }
        
        # Create visualizations
        self.create_distribution_plots(column)
        
        return analysis_results
    
    def analyze_all_variables(self) -> Dict[str, Any]:
        """
        Perform univariate analysis for all numerical variables.
        
        Returns:
        --------
        dict
            Complete analysis results for all variables
        """
        print("="*70)
        print("COMPREHENSIVE UNIVARIATE ANALYSIS")
        print("="*70)
        
        all_results = {
            'dataset_info': {
                'total_rows': len(self.df),
                'numerical_columns': len(self.numerical_columns),
                'columns_analyzed': self.numerical_columns.copy()
            },
            'variable_analyses': {},
            'summary_tables': {}
        }
        
        # Analyze each numerical variable
        for column in self.numerical_columns:
            try:
                all_results['variable_analyses'][column] = self.analyze_single_variable(column)
            except Exception as e:
                print(f"Error analyzing {column}: {str(e)}")
                all_results['variable_analyses'][column] = {'error': str(e)}
        
        # Generate summary tables
        all_results['summary_tables'] = self.generate_summary_tables(all_results['variable_analyses'])
        
        # Store results for later use
        self.results = all_results
        
        return all_results
    
    def generate_summary_tables(self, variable_analyses: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Generate summary tables from analysis results.
        
        Parameters:
        -----------
        variable_analyses : dict
            Analysis results for all variables
            
        Returns:
        --------
        dict
            Dictionary containing various summary tables
        """
        print("Generating summary tables...")
        
        # Descriptive Statistics Summary
        desc_stats_data = []
        normality_data = []
        skewness_data = []
        
        for column, analysis in variable_analyses.items():
            if 'error' in analysis:
                continue
                
            desc_stats = analysis['descriptive_statistics']
            if 'error' not in desc_stats:
                desc_stats_data.append({
                    'Variable': column,
                    'Count': desc_stats['count'],
                    'Missing%': desc_stats['missing_percentage'],
                    'Mean': desc_stats['mean'],
                    'Median': desc_stats['median'],
                    'Std': desc_stats['std'],
                    'Min': desc_stats['min'],
                    'Max': desc_stats['max'],
                    'Q1': desc_stats['percentile_25'],
                    'Q3': desc_stats['percentile_75'],
                    'IQR': desc_stats['iqr'],
                    'Skewness': desc_stats['skewness'],
                    'Kurtosis': desc_stats['kurtosis'],
                    'CV%': desc_stats['cv'],
                    'Outliers%': desc_stats['outlier_percentage']
                })
                
                # Skewness analysis
                skewness_data.append({
                    'Variable': column,
                    'Skewness': desc_stats['skewness'],
                    'Skewness_Interpretation': desc_stats['skewness_interpretation'],
                    'Excess_Kurtosis': desc_stats['excess_kurtosis'],
                    'Kurtosis_Interpretation': desc_stats['kurtosis_interpretation']
                })
            
            # Normality tests summary
            normality_tests = analysis['normality_tests']
            if 'error' not in normality_tests and 'overall_assessment' in normality_tests:
                assessment = normality_tests['overall_assessment']
                normality_row = {
                    'Variable': column,
                    'Tests_Passed': assessment['tests_passed'],
                    'Total_Tests': assessment['total_tests'],
                    'Normal_Percentage': assessment['percentage_normal'],
                    'Conclusion': assessment['conclusion']
                }
                
                # Add individual test results
                for test_name, test_result in normality_tests['tests'].items():
                    if 'error' not in test_result:
                        if 'p_value' in test_result:
                            normality_row[f'{test_name}_p'] = test_result['p_value']
                            normality_row[f'{test_name}_normal'] = test_result['is_normal']
                        elif 'is_normal_5pct' in test_result:
                            normality_row[f'{test_name}_normal'] = test_result['is_normal_5pct']
                
                normality_data.append(normality_row)
        
        # Create DataFrames
        summary_tables = {}
        
        if desc_stats_data:
            summary_tables['descriptive_statistics'] = pd.DataFrame(desc_stats_data)
            summary_tables['descriptive_statistics'] = summary_tables['descriptive_statistics'].round(4)
        
        if normality_data:
            summary_tables['normality_tests'] = pd.DataFrame(normality_data)
            
        if skewness_data:
            summary_tables['distribution_shapes'] = pd.DataFrame(skewness_data)
            summary_tables['distribution_shapes'] = summary_tables['distribution_shapes'].round(4)
        
        return summary_tables
    
    def print_analysis_summary(self) -> None:
        """Print a comprehensive summary of the analysis results."""
        
        if not self.results:
            print("No analysis results available. Run analyze_all_variables() first.")
            return
        
        print("\n" + "="*70)
        print("UNIVARIATE ANALYSIS SUMMARY")
        print("="*70)
        
        dataset_info = self.results['dataset_info']
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Total rows: {dataset_info['total_rows']:,}")
        print(f"   • Numerical columns analyzed: {dataset_info['numerical_columns']}")
        print(f"   • Visualizations saved to: {self.output_dir}/")
        
        # Summary of distribution shapes
        if 'distribution_shapes' in self.results['summary_tables']:
            shapes_df = self.results['summary_tables']['distribution_shapes']
            print(f"\n📈 DISTRIBUTION SHAPES:")
            skewness_counts = shapes_df['Skewness_Interpretation'].value_counts()
            for shape, count in skewness_counts.items():
                print(f"   • {shape}: {count} variables")
        
        # Normality test summary
        if 'normality_tests' in self.results['summary_tables']:
            norm_df = self.results['summary_tables']['normality_tests']
            print(f"\n🔍 NORMALITY ASSESSMENT:")
            conclusion_counts = norm_df['Conclusion'].value_counts()
            for conclusion, count in conclusion_counts.items():
                print(f"   • {conclusion}: {count} variables")
        
        # Identify problematic variables
        if 'descriptive_statistics' in self.results['summary_tables']:
            desc_df = self.results['summary_tables']['descriptive_statistics']
            
            print(f"\n⚠️  VARIABLES REQUIRING ATTENTION:")
            # High missing data
            high_missing = desc_df[desc_df['Missing%'] > 10]
            if not high_missing.empty:
                print("   High missing data (>10%):")
                for _, row in high_missing.iterrows():
                    print(f"     • {row['Variable']}: {row['Missing%']:.1f}% missing")
            
            # Highly skewed variables
            highly_skewed = desc_df[abs(desc_df['Skewness']) > 2]
            if not highly_skewed.empty:
                print("   Highly skewed variables (|skewness| > 2):")
                for _, row in highly_skewed.iterrows():
                    print(f"     • {row['Variable']}: skewness = {row['Skewness']:.2f}")
            
            # High outlier percentage
            high_outliers = desc_df[desc_df['Outliers%'] > 5]
            if not high_outliers.empty:
                print("   High outlier percentage (>5%):")
                for _, row in high_outliers.iterrows():
                    print(f"     • {row['Variable']}: {row['Outliers%']:.1f}% outliers")
        
        print(f"\n💡 RECOMMENDATIONS:")
        self._generate_analysis_recommendations()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
    
    def _generate_analysis_recommendations(self) -> None:
        """Generate recommendations based on analysis results."""
        
        recommendations = []
        
        if 'descriptive_statistics' in self.results['summary_tables']:
            desc_df = self.results['summary_tables']['descriptive_statistics']
            
            # Check for highly skewed variables
            highly_skewed = desc_df[abs(desc_df['Skewness']) > 2]
            if not highly_skewed.empty:
                recommendations.append(f"Consider transformations for {len(highly_skewed)} highly skewed variables")
            
            # Check for high coefficient of variation
            high_cv = desc_df[desc_df['CV%'] > 100]
            if not high_cv.empty:
                recommendations.append(f"High variability detected in {len(high_cv)} variables - consider standardization")
            
            # Check normality
            if 'normality_tests' in self.results['summary_tables']:
                norm_df = self.results['summary_tables']['normality_tests']
                non_normal = norm_df[norm_df['Normal_Percentage'] < 50]
                if not non_normal.empty:
                    recommendations.append(f"{len(non_normal)} variables are not normally distributed - consider non-parametric methods")
        
        if not recommendations:
            recommendations.append("Data appears to be in good shape for analysis")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

def comprehensive_univariate_analysis(df: pd.DataFrame, 
                                    output_dir: str = 'univariate_analysis_plots') -> Dict[str, Any]:
    """
    Perform comprehensive univariate analysis on all numerical variables.
    
    This function provides extensive statistical analysis including:
    - Descriptive statistics (mean, median, std, skewness, kurtosis, percentiles)
    - Multiple normality tests (Shapiro-Wilk, Anderson-Darling, D'Agostino-Pearson, etc.)
    - Professional distribution visualizations (histograms, box plots, Q-Q plots)
    - Outlier detection and analysis
    - Distribution shape analysis
    - Summary tables and recommendations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    output_dir : str, default 'univariate_analysis_plots'
        Directory to save visualization plots
        
    Returns:
    --------
    dict
        Comprehensive analysis results including:
        - Descriptive statistics for all variables
        - Normality test results
        - Distribution shape analysis
        - Summary tables
        - Visualizations saved to files
        
    Generated Visualizations:
    -------------------------
    For each numerical variable:
    - distribution_[variable].png: 4-panel plot (histogram+KDE, box plot, Q-Q plot, violin plot)
    - detailed_histogram_[variable].png: Detailed histogram with statistical overlays
    """
    
    # Initialize analyzer
    analyzer = UnivariateAnalyzer(df, output_dir)
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_variables()
    
    # Print summary
    analyzer.print_analysis_summary()
    
    return results

# Example usage
if __name__ == "__main__":
    # Load wine dataset for testing
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Perform comprehensive univariate analysis
    analysis_results = comprehensive_univariate_analysis(df)