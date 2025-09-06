import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EmpiricalCovariance
import warnings
import os
from typing import Dict, List, Tuple, Any, Union
warnings.filterwarnings('ignore')

class AdvancedOutlierDetector:
    """
    Advanced outlier detection system for wine dataset analysis.
    
    Implements multiple outlier detection methods, consensus scoring,
    domain-specific flagging, and treatment recommendations.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'outlier_analysis_plots'):
        """
        Initialize the advanced outlier detector.
        
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
        self.outlier_scores = {}
        
        # Wine chemistry parameter ranges for domain-specific detection
        self.wine_chemistry_ranges = {
            'pH': {'extreme_min': 2.5, 'extreme_max': 4.5, 'typical_min': 3.0, 'typical_max': 3.8},
            'Alcohol': {'extreme_min': 5.0, 'extreme_max': 20.0, 'typical_min': 8.0, 'typical_max': 16.0},
            'FixedAcidity': {'extreme_min': 2.0, 'extreme_max': 18.0, 'typical_min': 4.0, 'typical_max': 12.0},
            'VolatileAcidity': {'extreme_min': 0.0, 'extreme_max': 2.0, 'typical_min': 0.1, 'typical_max': 1.2},
            'ResidualSugar': {'extreme_min': 0.0, 'extreme_max': 300.0, 'typical_min': 0.5, 'typical_max': 150.0},
            'Density': {'extreme_min': 0.980, 'extreme_max': 1.020, 'typical_min': 0.985, 'typical_max': 1.005}
        }
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _identify_numerical_columns(self) -> List[str]:
        """Identify numerical columns suitable for outlier analysis."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove index-like columns
        numerical_cols = [col for col in numerical_cols if not col.upper().startswith('INDEX')]
        return numerical_cols
    
    def detect_iqr_outliers(self, column: str, multiplier: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        multiplier : float, default 1.5
            IQR multiplier for outlier boundary
            
        Returns:
        --------
        dict
            IQR outlier detection results
        """
        data = self.df[column].dropna()
        
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outliers = data[outlier_mask]
        
        return {
            'method': 'IQR',
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.values.tolist(),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'multiplier': multiplier
        }
    
    def detect_zscore_outliers(self, column: str, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        threshold : float, default 3.0
            Z-score threshold for outlier detection
            
        Returns:
        --------
        dict
            Z-score outlier detection results
        """
        data = self.df[column].dropna()
        
        mean_val = data.mean()
        std_val = data.std()
        
        if std_val == 0:
            return {
                'method': 'Z-Score',
                'outlier_indices': [],
                'outlier_values': [],
                'outlier_count': 0,
                'outlier_percentage': 0,
                'error': 'Zero standard deviation - no variability in data'
            }
        
        z_scores = np.abs((data - mean_val) / std_val)
        outlier_mask = z_scores > threshold
        outliers = data[outlier_mask]
        
        return {
            'method': 'Z-Score',
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.values.tolist(),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'z_scores': z_scores[outlier_mask].values.tolist(),
            'threshold': threshold,
            'mean': mean_val,
            'std': std_val
        }
    
    def detect_modified_zscore_outliers(self, column: str, threshold: float = 3.5) -> Dict[str, Any]:
        """
        Detect outliers using Modified Z-score method (robust to extreme values).
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        threshold : float, default 3.5
            Modified Z-score threshold for outlier detection
            
        Returns:
        --------
        dict
            Modified Z-score outlier detection results
        """
        data = self.df[column].dropna()
        
        median_val = data.median()
        mad = np.median(np.abs(data - median_val))  # Median Absolute Deviation
        
        if mad == 0:
            # Use a robust alternative when MAD is zero
            mad = np.median(np.abs(data - median_val)) + 1e-10
        
        # Modified Z-score formula
        modified_z_scores = 0.6745 * (data - median_val) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outliers = data[outlier_mask]
        
        return {
            'method': 'Modified Z-Score',
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.values.tolist(),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'modified_z_scores': modified_z_scores[outlier_mask].values.tolist(),
            'threshold': threshold,
            'median': median_val,
            'mad': mad
        }
    
    def detect_isolation_forest_outliers(self, column: str, contamination: float = 0.1, 
                                       random_state: int = 42) -> Dict[str, Any]:
        """
        Detect outliers using Isolation Forest method.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        contamination : float, default 0.1
            Expected proportion of outliers
        random_state : int, default 42
            Random state for reproducibility
            
        Returns:
        --------
        dict
            Isolation Forest outlier detection results
        """
        data = self.df[column].dropna()
        
        if len(data) < 10:
            return {
                'method': 'Isolation Forest',
                'outlier_indices': [],
                'outlier_values': [],
                'outlier_count': 0,
                'outlier_percentage': 0,
                'error': 'Insufficient data for Isolation Forest'
            }
        
        # Reshape data for sklearn
        X = data.values.reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Get outliers (labeled as -1)
        outlier_mask = outlier_labels == -1
        outliers = data[outlier_mask]
        
        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(X)
        
        return {
            'method': 'Isolation Forest',
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.values.tolist(),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'anomaly_scores': anomaly_scores[outlier_mask].tolist(),
            'contamination': contamination,
            'model': iso_forest
        }
    
    def detect_multivariate_outliers(self, method: str = 'mahalanobis', 
                                   threshold_percentile: float = 97.5) -> Dict[str, Any]:
        """
        Detect multivariate outliers using Mahalanobis distance or Isolation Forest.
        
        Parameters:
        -----------
        method : str, default 'mahalanobis'
            Method to use ('mahalanobis' or 'isolation_forest')
        threshold_percentile : float, default 97.5
            Percentile threshold for Mahalanobis distance
            
        Returns:
        --------
        dict
            Multivariate outlier detection results
        """
        print(f"Detecting multivariate outliers using {method}...")
        
        # Get clean numerical data
        clean_df = self.df[self.numerical_columns].dropna()
        
        if len(clean_df) < 10:
            return {
                'method': f'Multivariate {method}',
                'error': 'Insufficient data for multivariate outlier detection'
            }
        
        if method == 'mahalanobis':
            try:
                # Calculate covariance matrix
                cov_matrix = np.cov(clean_df.T)
                inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Use pseudoinverse for stability
                
                # Calculate mean vector
                mean_vector = clean_df.mean().values
                
                # Calculate Mahalanobis distances
                mahal_distances = []
                for idx, row in clean_df.iterrows():
                    diff = row.values - mean_vector
                    mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                    mahal_distances.append(mahal_dist)
                
                mahal_distances = np.array(mahal_distances)
                
                # Determine threshold using chi-square distribution
                # Degrees of freedom = number of variables
                df_chi2 = len(self.numerical_columns)
                chi2_threshold = chi2.ppf(threshold_percentile / 100, df_chi2)
                
                # Alternative: use percentile-based threshold
                percentile_threshold = np.percentile(mahal_distances, threshold_percentile)
                threshold = max(chi2_threshold, percentile_threshold)
                
                # Identify outliers
                outlier_mask = mahal_distances > threshold
                outlier_indices = clean_df.index[outlier_mask].tolist()
                
                return {
                    'method': 'Multivariate Mahalanobis',
                    'outlier_indices': outlier_indices,
                    'outlier_count': len(outlier_indices),
                    'outlier_percentage': (len(outlier_indices) / len(clean_df)) * 100,
                    'mahalanobis_distances': mahal_distances.tolist(),
                    'threshold': threshold,
                    'chi2_threshold': chi2_threshold,
                    'percentile_threshold': percentile_threshold,
                    'degrees_of_freedom': df_chi2
                }
                
            except Exception as e:
                return {
                    'method': 'Multivariate Mahalanobis',
                    'error': f'Mahalanobis calculation failed: {str(e)}'
                }
        
        elif method == 'isolation_forest':
            # Multivariate Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(clean_df)
            
            outlier_mask = outlier_labels == -1
            outlier_indices = clean_df.index[outlier_mask].tolist()
            
            return {
                'method': 'Multivariate Isolation Forest',
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': (len(outlier_indices) / len(clean_df)) * 100,
                'anomaly_scores': iso_forest.decision_function(clean_df)[outlier_mask].tolist()
            }
        
        else:
            return {'error': f'Unknown method: {method}'}
    
    def detect_domain_specific_outliers(self) -> Dict[str, Any]:
        """
        Detect wine chemistry domain-specific outliers.
        
        Returns:
        --------
        dict
            Domain-specific outlier detection results
        """
        print("Detecting domain-specific wine chemistry outliers...")
        
        domain_outliers = {
            'chemically_impossible': [],
            'extremely_rare': [],
            'suspicious_combinations': [],
            'parameter_violations': {}
        }
        
        # Check individual parameter violations
        for param, ranges in self.wine_chemistry_ranges.items():
            if param in self.df.columns:
                data = self.df[param].dropna()
                
                # Chemically impossible values
                impossible_mask = (data < ranges['extreme_min']) | (data > ranges['extreme_max'])
                impossible_indices = data[impossible_mask].index.tolist()
                
                # Extremely rare but possible values
                rare_mask = ((data >= ranges['extreme_min']) & (data < ranges['typical_min'])) | \
                           ((data > ranges['typical_max']) & (data <= ranges['extreme_max']))
                rare_indices = data[rare_mask].index.tolist()
                
                domain_outliers['parameter_violations'][param] = {
                    'impossible_indices': impossible_indices,
                    'impossible_count': len(impossible_indices),
                    'rare_indices': rare_indices,
                    'rare_count': len(rare_indices)
                }
                
                domain_outliers['chemically_impossible'].extend(impossible_indices)
                domain_outliers['extremely_rare'].extend(rare_indices)
        
        # Check suspicious combinations
        suspicious_combinations = []
        
        # High alcohol with very high residual sugar (unusual combination)
        if 'Alcohol' in self.df.columns and 'ResidualSugar' in self.df.columns:
            high_alcohol_high_sugar = self.df[
                (self.df['Alcohol'] > 15.0) & 
                (self.df['ResidualSugar'] > 100.0)
            ].index.tolist()
            suspicious_combinations.extend(high_alcohol_high_sugar)
        
        # Very low fixed acidity with very high pH
        if 'FixedAcidity' in self.df.columns and 'pH' in self.df.columns:
            low_acid_high_ph = self.df[
                (self.df['FixedAcidity'] < 4.0) & 
                (self.df['pH'] > 4.0)
            ].index.tolist()
            suspicious_combinations.extend(low_acid_high_ph)
        
        # Total SO2 less than Free SO2 (impossible)
        if 'TotalSulfurDioxide' in self.df.columns and 'FreeSulfurDioxide' in self.df.columns:
            impossible_so2 = self.df[
                self.df['TotalSulfurDioxide'] < self.df['FreeSulfurDioxide']
            ].index.tolist()
            suspicious_combinations.extend(impossible_so2)
        
        domain_outliers['suspicious_combinations'] = list(set(suspicious_combinations))
        domain_outliers['chemically_impossible'] = list(set(domain_outliers['chemically_impossible']))
        domain_outliers['extremely_rare'] = list(set(domain_outliers['extremely_rare']))
        
        return domain_outliers
    
    def create_consensus_outlier_scores(self, methods_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create consensus outlier scores by combining multiple detection methods.
        
        Parameters:
        -----------
        methods_results : dict
            Results from multiple outlier detection methods
            
        Returns:
        --------
        dict
            Consensus outlier scoring results
        """
        print("Creating consensus outlier scores...")
        
        # Initialize consensus scores
        all_indices = self.df.index.tolist()
        consensus_scores = pd.Series(0, index=all_indices)
        method_counts = pd.Series(0, index=all_indices)
        
        # Weight different methods
        method_weights = {
            'IQR': 1.0,
            'Z-Score': 1.0,
            'Modified Z-Score': 1.2,  # More robust, higher weight
            'Isolation Forest': 1.1,
            'Multivariate Mahalanobis': 1.5,  # Multivariate gets higher weight
            'Multivariate Isolation Forest': 1.3,
            'Domain-Specific': 2.0  # Highest weight for domain knowledge
        }
        
        # Aggregate scores from all methods
        for variable, variable_results in methods_results.items():
            if isinstance(variable_results, dict):
                for method_name, method_results in variable_results.items():
                    if 'outlier_indices' in method_results and 'error' not in method_results:
                        outlier_indices = method_results['outlier_indices']
                        weight = method_weights.get(method_name, 1.0)
                        
                        # Add weighted score for each outlier
                        for idx in outlier_indices:
                            if idx in consensus_scores.index:
                                consensus_scores[idx] += weight
                                method_counts[idx] += 1
        
        # Handle domain-specific outliers separately
        if 'domain_specific' in methods_results:
            domain_results = methods_results['domain_specific']
            weight = method_weights.get('Domain-Specific', 2.0)
            
            # Add scores for different types of domain violations
            for violation_type, indices in domain_results.items():
                if isinstance(indices, list):
                    type_weight = weight * (2.0 if violation_type == 'chemically_impossible' else 1.0)
                    for idx in indices:
                        if idx in consensus_scores.index:
                            consensus_scores[idx] += type_weight
                            method_counts[idx] += 1
        
        # Create outlier severity categories
        outlier_categories = pd.Series('Normal', index=all_indices)
        
        # Define thresholds based on consensus scores
        high_threshold = np.percentile(consensus_scores[consensus_scores > 0], 90) if (consensus_scores > 0).any() else 0
        moderate_threshold = np.percentile(consensus_scores[consensus_scores > 0], 75) if (consensus_scores > 0).any() else 0
        
        outlier_categories[consensus_scores >= high_threshold] = 'Severe Outlier'
        outlier_categories[(consensus_scores >= moderate_threshold) & (consensus_scores < high_threshold)] = 'Moderate Outlier'
        outlier_categories[(consensus_scores > 0) & (consensus_scores < moderate_threshold)] = 'Mild Outlier'
        
        return {
            'consensus_scores': consensus_scores,
            'method_counts': method_counts,
            'outlier_categories': outlier_categories,
            'severe_outliers': consensus_scores[consensus_scores >= high_threshold].index.tolist(),
            'moderate_outliers': consensus_scores[(consensus_scores >= moderate_threshold) & (consensus_scores < high_threshold)].index.tolist(),
            'mild_outliers': consensus_scores[(consensus_scores > 0) & (consensus_scores < moderate_threshold)].index.tolist(),
            'high_threshold': high_threshold,
            'moderate_threshold': moderate_threshold,
            'total_outliers': (consensus_scores > 0).sum(),
            'outlier_percentage': ((consensus_scores > 0).sum() / len(consensus_scores)) * 100
        }
    
    def generate_outlier_visualizations(self, consensus_results: Dict[str, Any],
                                      methods_results: Dict[str, Dict]) -> None:
        """
        Generate comprehensive outlier visualizations.
        
        Parameters:
        -----------
        consensus_results : dict
            Consensus outlier scoring results
        methods_results : dict
            Results from individual detection methods
        """
        print("Generating outlier visualizations...")
        
        # Visualization 1: Consensus outlier scores distribution
        self._create_consensus_score_plot(consensus_results)
        
        # Visualization 2: Method comparison heatmap
        self._create_method_comparison_heatmap(methods_results, consensus_results)
        
        # Visualization 3: Box plots with outlier annotations
        self._create_annotated_boxplots(consensus_results)
        
        # Visualization 4: Multivariate outlier scatter plots
        self._create_multivariate_scatter_plots(consensus_results, methods_results)
        
        # Visualization 5: Outlier method agreement chart
        self._create_method_agreement_chart(consensus_results)
    
    def _create_consensus_score_plot(self, consensus_results: Dict[str, Any]) -> None:
        """Create consensus outlier scores visualization."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        consensus_scores = consensus_results['consensus_scores']
        
        # Plot 1: Histogram of consensus scores
        ax1.hist(consensus_scores[consensus_scores > 0], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(consensus_results['moderate_threshold'], color='orange', 
                   linestyle='--', label=f'Moderate threshold ({consensus_results["moderate_threshold"]:.2f})')
        ax1.axvline(consensus_results['high_threshold'], color='red', 
                   linestyle='--', label=f'Severe threshold ({consensus_results["high_threshold"]:.2f})')
        ax1.set_xlabel('Consensus Outlier Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Consensus Outlier Scores')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Outlier categories pie chart
        category_counts = consensus_results['outlier_categories'].value_counts()
        colors = ['lightgreen', 'yellow', 'orange', 'red'][:len(category_counts)]
        
        ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Outlier Severity Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/consensus_outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_method_comparison_heatmap(self, methods_results: Dict[str, Dict],
                                        consensus_results: Dict[str, Any]) -> None:
        """Create heatmap comparing different outlier detection methods."""
        
        # Collect method results for each variable
        method_data = []
        variables = []
        methods = []
        
        for variable, variable_results in methods_results.items():
            if variable != 'domain_specific' and isinstance(variable_results, dict):
                variables.append(variable)
                variable_methods = []
                variable_percentages = []
                
                for method_name, method_results in variable_results.items():
                    if 'outlier_percentage' in method_results:
                        variable_methods.append(method_name)
                        variable_percentages.append(method_results['outlier_percentage'])
                
                method_data.append(variable_percentages)
                if not methods:  # First iteration
                    methods = variable_methods
        
        if method_data and methods:
            # Create DataFrame
            comparison_df = pd.DataFrame(method_data, index=variables, columns=methods)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(comparison_df, annot=True, fmt='.1f', cmap='Reds', 
                       cbar_kws={'label': 'Outlier Percentage (%)'})
            plt.title('Outlier Detection Methods Comparison\n(Percentage of outliers detected by each method)')
            plt.xlabel('Detection Method')
            plt.ylabel('Variable')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_annotated_boxplots(self, consensus_results: Dict[str, Any]) -> None:
        """Create box plots with outlier annotations."""
        
        # Select key variables for visualization
        key_variables = self.numerical_columns[:8]  # Limit to 8 variables for readability
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        consensus_scores = consensus_results['consensus_scores']
        
        for i, variable in enumerate(key_variables):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = self.df[variable].dropna()
            
            # Create box plot
            bp = ax.boxplot(data, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # Annotate severe outliers
            severe_outliers = consensus_results['severe_outliers']
            severe_outlier_data = []
            
            for idx in severe_outliers:
                if idx in data.index and not pd.isna(self.df.loc[idx, variable]):
                    severe_outlier_data.append(self.df.loc[idx, variable])
            
            # Plot severe outliers as red points
            if severe_outlier_data:
                ax.scatter([1] * len(severe_outlier_data), severe_outlier_data, 
                          color='red', s=50, alpha=0.7, zorder=5)
            
            ax.set_title(f'{variable}\n({len(severe_outlier_data)} severe outliers)')
            ax.set_ylabel(variable)
            ax.grid(alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(key_variables), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Box Plots with Severe Outliers Highlighted (Red Points)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/annotated_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_multivariate_scatter_plots(self, consensus_results: Dict[str, Any],
                                         methods_results: Dict[str, Dict]) -> None:
        """Create scatter plots for multivariate outlier visualization."""
        
        # Select two most correlated variables for scatter plot
        if len(self.numerical_columns) >= 2:
            var1, var2 = self.numerical_columns[0], self.numerical_columns[1]
            
            # Try to find TARGET variable and a chemistry variable
            if 'TARGET' in self.numerical_columns:
                var1 = 'TARGET'
                chemistry_vars = [col for col in self.numerical_columns if col != 'TARGET']
                if chemistry_vars:
                    var2 = chemistry_vars[0]
            
            plt.figure(figsize=(12, 8))
            
            # Plot normal points
            normal_mask = consensus_results['outlier_categories'] == 'Normal'
            plt.scatter(self.df.loc[normal_mask, var1], self.df.loc[normal_mask, var2],
                       alpha=0.6, color='lightblue', s=30, label='Normal')
            
            # Plot outliers with different colors by severity
            for severity, color in [('Mild Outlier', 'yellow'), ('Moderate Outlier', 'orange'), 
                                  ('Severe Outlier', 'red')]:
                severity_mask = consensus_results['outlier_categories'] == severity
                if severity_mask.any():
                    plt.scatter(self.df.loc[severity_mask, var1], self.df.loc[severity_mask, var2],
                               alpha=0.8, color=color, s=60, label=severity, edgecolors='black')
            
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'Multivariate Outlier Visualization: {var1} vs {var2}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/multivariate_outlier_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_method_agreement_chart(self, consensus_results: Dict[str, Any]) -> None:
        """Create chart showing method agreement levels."""
        
        method_counts = consensus_results['method_counts']
        
        plt.figure(figsize=(10, 6))
        
        # Count how many methods agree on each outlier
        agreement_counts = method_counts[method_counts > 0].value_counts().sort_index()
        
        bars = plt.bar(range(1, len(agreement_counts) + 1), agreement_counts.values, 
                      alpha=0.7, edgecolor='black')
        
        # Color bars based on agreement level
        colors = ['lightcoral', 'orange', 'yellow', 'lightgreen', 'green']
        for i, bar in enumerate(bars):
            agreement_level = i + 1
            color_idx = min(agreement_level - 1, len(colors) - 1)
            bar.set_color(colors[color_idx])
        
        plt.xlabel('Number of Methods in Agreement')
        plt.ylabel('Number of Data Points')
        plt.title('Outlier Detection Method Agreement\n(How many methods agree on each outlier)')
        plt.xticks(range(1, len(agreement_counts) + 1))
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/method_agreement_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_treatment_recommendations(self, consensus_results: Dict[str, Any],
                                         methods_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate outlier treatment recommendations.
        
        Parameters:
        -----------
        consensus_results : dict
            Consensus outlier scoring results
        methods_results : dict
            Individual method results
            
        Returns:
        --------
        dict
            Treatment recommendations
        """
        print("Generating outlier treatment recommendations...")
        
        recommendations = {
            'immediate_action': [],
            'review_recommended': [],
            'monitoring_suggested': [],
            'treatment_strategies': {},
            'variable_specific_actions': {}
        }
        
        # Categorize outliers by severity and recommend actions
        severe_outliers = consensus_results['severe_outliers']
        moderate_outliers = consensus_results['moderate_outliers']
        mild_outliers = consensus_results['mild_outliers']
        
        # Immediate action for severe outliers
        if severe_outliers:
            recommendations['immediate_action'].append(
                f"Remove or investigate {len(severe_outliers)} severe outliers before ML modeling"
            )
            recommendations['immediate_action'].append(
                "Verify data collection methods for severe outlier cases"
            )
        
        # Review recommendations for moderate outliers
        if moderate_outliers:
            recommendations['review_recommended'].append(
                f"Review {len(moderate_outliers)} moderate outliers - consider capping or transformation"
            )
            recommendations['review_recommended'].append(
                "Use robust scaling methods if keeping moderate outliers"
            )
        
        # Monitoring for mild outliers
        if mild_outliers:
            recommendations['monitoring_suggested'].append(
                f"Monitor {len(mild_outliers)} mild outliers during model validation"
            )
        
        # Treatment strategies
        outlier_percentage = consensus_results['outlier_percentage']
        
        if outlier_percentage > 20:
            recommendations['treatment_strategies']['high_outlier_rate'] = [
                "High outlier rate detected (>20%)",
                "Consider robust statistical methods",
                "Investigate data preprocessing issues",
                "Use ensemble methods that handle outliers well"
            ]
        elif outlier_percentage > 10:
            recommendations['treatment_strategies']['moderate_outlier_rate'] = [
                "Moderate outlier rate (10-20%)",
                "Consider outlier-robust transformations",
                "Use cross-validation to assess impact",
                "Compare models with and without outlier treatment"
            ]
        else:
            recommendations['treatment_strategies']['low_outlier_rate'] = [
                "Low outlier rate (<10%)",
                "Standard outlier treatments should work well",
                "Consider keeping outliers with robust methods"
            ]
        
        # Variable-specific recommendations
        for variable in self.numerical_columns:
            if variable in methods_results:
                var_results = methods_results[variable]
                
                # Find method with highest outlier detection rate
                max_outlier_rate = 0
                problematic_method = None
                
                for method_name, method_result in var_results.items():
                    if 'outlier_percentage' in method_result:
                        if method_result['outlier_percentage'] > max_outlier_rate:
                            max_outlier_rate = method_result['outlier_percentage']
                            problematic_method = method_name
                
                if max_outlier_rate > 15:
                    recommendations['variable_specific_actions'][variable] = {
                        'issue': f'High outlier rate: {max_outlier_rate:.1f}%',
                        'detected_by': problematic_method,
                        'recommendations': [
                            'Consider log transformation if right-skewed',
                            'Use robust scaling (median and MAD)',
                            'Investigate for data entry errors',
                            'Consider capping at 95th/99th percentiles'
                        ]
                    }
        
        return recommendations
    
    def create_before_after_framework(self, consensus_results: Dict[str, Any],
                                    treatment_method: str = 'cap_percentiles') -> Dict[str, Any]:
        """
        Create framework for comparing data before and after outlier treatment.
        
        Parameters:
        -----------
        consensus_results : dict
            Consensus outlier results
        treatment_method : str
            Treatment method to apply ('remove', 'cap_percentiles', 'robust_scale')
            
        Returns:
        --------
        dict
            Before/after comparison results
        """
        print(f"Creating before/after comparison using {treatment_method}...")
        
        # Create treated dataset
        treated_df = self.df.copy()
        treatment_summary = {}
        
        if treatment_method == 'remove':
            # Remove severe outliers
            severe_outliers = consensus_results['severe_outliers']
            treated_df = treated_df.drop(severe_outliers)
            treatment_summary['method'] = 'Removed severe outliers'
            treatment_summary['rows_removed'] = len(severe_outliers)
            
        elif treatment_method == 'cap_percentiles':
            # Cap at 1st and 99th percentiles
            for column in self.numerical_columns:
                if column in treated_df.columns:
                    p1 = treated_df[column].quantile(0.01)
                    p99 = treated_df[column].quantile(0.99)
                    
                    original_extreme_count = ((treated_df[column] < p1) | (treated_df[column] > p99)).sum()
                    
                    treated_df[column] = treated_df[column].clip(lower=p1, upper=p99)
                    
                    treatment_summary[column] = {
                        'lower_cap': p1,
                        'upper_cap': p99,
                        'values_capped': original_extreme_count
                    }
            
            treatment_summary['method'] = 'Capped at 1st and 99th percentiles'
            
        elif treatment_method == 'robust_scale':
            # Apply robust scaling (median and MAD)
            scaler = RobustScaler()
            treated_df[self.numerical_columns] = scaler.fit_transform(treated_df[self.numerical_columns])
            treatment_summary['method'] = 'Applied robust scaling (median/MAD)'
            treatment_summary['scaler'] = scaler
        
        # Compare statistics before and after
        comparison = {}
        
        for column in self.numerical_columns:
            if column in self.df.columns and column in treated_df.columns:
                before_stats = {
                    'count': self.df[column].count(),
                    'mean': self.df[column].mean(),
                    'median': self.df[column].median(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max(),
                    'skewness': self.df[column].skew(),
                    'kurtosis': self.df[column].kurtosis()
                }
                
                after_stats = {
                    'count': treated_df[column].count(),
                    'mean': treated_df[column].mean(),
                    'median': treated_df[column].median(),
                    'std': treated_df[column].std(),
                    'min': treated_df[column].min(),
                    'max': treated_df[column].max(),
                    'skewness': treated_df[column].skew(),
                    'kurtosis': treated_df[column].kurtosis()
                }
                
                comparison[column] = {
                    'before': before_stats,
                    'after': after_stats,
                    'changes': {
                        'mean_change': after_stats['mean'] - before_stats['mean'],
                        'std_reduction': before_stats['std'] - after_stats['std'],
                        'skewness_improvement': abs(before_stats['skewness']) - abs(after_stats['skewness']),
                        'range_reduction': (before_stats['max'] - before_stats['min']) - 
                                         (after_stats['max'] - after_stats['min'])
                    }
                }
        
        # Create before/after visualization
        self._create_before_after_plots(comparison, treatment_method)
        
        return {
            'treated_dataframe': treated_df,
            'treatment_summary': treatment_summary,
            'statistical_comparison': comparison,
            'treatment_method': treatment_method
        }
    
    def _create_before_after_plots(self, comparison: Dict[str, Any], treatment_method: str) -> None:
        """Create before/after comparison plots."""
        
        # Select key variables for visualization
        key_variables = list(comparison.keys())[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, variable in enumerate(key_variables):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            before_stats = comparison[variable]['before']
            after_stats = comparison[variable]['after']
            
            # Create before/after bar chart for key statistics
            stats = ['mean', 'median', 'std', 'min', 'max']
            before_values = [before_stats[stat] for stat in stats]
            after_values = [after_stats[stat] for stat in stats]
            
            x = np.arange(len(stats))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, before_values, width, label='Before', alpha=0.7)
            bars2 = ax.bar(x + width/2, after_values, width, label='After', alpha=0.7)
            
            ax.set_xlabel('Statistics')
            ax.set_ylabel('Value')
            ax.set_title(f'{variable} - Before vs After Treatment')
            ax.set_xticks(x)
            ax.set_xticklabels(stats)
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Remove empty subplots
        for i in range(len(key_variables), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Before vs After Outlier Treatment: {treatment_method}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/before_after_comparison_{treatment_method}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def comprehensive_outlier_analysis(df: pd.DataFrame, 
                                 output_dir: str = 'outlier_analysis_plots') -> Dict[str, Any]:
    """
    Perform comprehensive outlier analysis on wine dataset.
    
    This function implements multiple outlier detection methods, creates consensus
    scores, generates treatment recommendations, and provides before/after analysis.
    
    Outlier Detection Methods:
    -------------------------
    1. IQR Method: Traditional quartile-based outlier detection
    2. Z-Score Method: Standard deviation-based detection (threshold = 3.0)
    3. Modified Z-Score: Robust median-based detection (threshold = 3.5)  
    4. Isolation Forest: Machine learning-based anomaly detection
    5. Multivariate Mahalanobis: Multi-dimensional distance-based detection
    6. Domain-Specific: Wine chemistry knowledge-based detection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wine dataset to analyze
    output_dir : str, default 'outlier_analysis_plots'
        Directory to save visualization plots
        
    Returns:
    --------
    dict
        Comprehensive outlier analysis results including:
        - Individual method results for each variable
        - Consensus outlier scores and categories
        - Treatment recommendations
        - Before/after comparison framework
        - Professional visualizations
        
    Generated Visualizations:
    -------------------------
    - consensus_outlier_analysis.png: Score distribution and severity breakdown
    - method_comparison_heatmap.png: Comparison of detection methods
    - annotated_boxplots.png: Box plots with severe outliers highlighted
    - multivariate_outlier_scatter.png: 2D scatter plot of multivariate outliers
    - method_agreement_chart.png: Agreement levels between methods
    - before_after_comparison_[method].png: Treatment effect visualization
    """
    
    print("="*70)
    print("COMPREHENSIVE ADVANCED OUTLIER ANALYSIS")
    print("="*70)
    
    # Initialize detector
    detector = AdvancedOutlierDetector(df, output_dir)
    
    # Run outlier detection for each variable using multiple methods
    print("\n🔍 RUNNING MULTIPLE OUTLIER DETECTION METHODS")
    print("-" * 50)
    
    all_methods_results = {}
    
    for variable in detector.numerical_columns:
        print(f"Analyzing {variable}...")
        
        variable_results = {
            'IQR': detector.detect_iqr_outliers(variable),
            'Z-Score': detector.detect_zscore_outliers(variable),
            'Modified Z-Score': detector.detect_modified_zscore_outliers(variable),
            'Isolation Forest': detector.detect_isolation_forest_outliers(variable)
        }
        
        all_methods_results[variable] = variable_results
    
    # Multivariate outlier detection
    print("\n🎯 MULTIVARIATE OUTLIER DETECTION")
    print("-" * 50)
    
    multivariate_mahal = detector.detect_multivariate_outliers('mahalanobis')
    multivariate_iso = detector.detect_multivariate_outliers('isolation_forest')
    
    all_methods_results['multivariate'] = {
        'Mahalanobis': multivariate_mahal,
        'Isolation Forest': multivariate_iso
    }
    
    # Domain-specific outlier detection
    print("\n🍷 DOMAIN-SPECIFIC WINE CHEMISTRY ANALYSIS")
    print("-" * 50)
    
    domain_specific = detector.detect_domain_specific_outliers()
    all_methods_results['domain_specific'] = domain_specific
    
    # Create consensus scores
    print("\n🤝 CREATING CONSENSUS OUTLIER SCORES")
    print("-" * 50)
    
    consensus_results = detector.create_consensus_outlier_scores(all_methods_results)
    
    # Generate visualizations
    print("\n📊 GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    detector.generate_outlier_visualizations(consensus_results, all_methods_results)
    
    # Generate treatment recommendations
    print("\n💊 GENERATING TREATMENT RECOMMENDATIONS")
    print("-" * 50)
    
    treatment_recommendations = detector.generate_treatment_recommendations(
        consensus_results, all_methods_results)
    
    # Create before/after comparison
    print("\n🔄 CREATING BEFORE/AFTER ANALYSIS")
    print("-" * 50)
    
    before_after_analysis = detector.create_before_after_framework(
        consensus_results, treatment_method='cap_percentiles')
    
    # Compile final results
    final_results = {
        'individual_methods': all_methods_results,
        'consensus_analysis': consensus_results,
        'treatment_recommendations': treatment_recommendations,
        'before_after_analysis': before_after_analysis,
        'dataset_summary': {
            'total_rows': len(df),
            'numerical_variables': len(detector.numerical_columns),
            'total_outliers': consensus_results['total_outliers'],
            'outlier_percentage': consensus_results['outlier_percentage'],
            'severe_outliers': len(consensus_results['severe_outliers']),
            'moderate_outliers': len(consensus_results['moderate_outliers']),
            'mild_outliers': len(consensus_results['mild_outliers'])
        }
    }
    
    # Print summary
    print(f"\n📊 OUTLIER ANALYSIS SUMMARY:")
    print(f"   • Total data points: {len(df):,}")
    print(f"   • Variables analyzed: {len(detector.numerical_columns)}")
    print(f"   • Detection methods used: 6")
    print(f"   • Total outliers detected: {consensus_results['total_outliers']:,} ({consensus_results['outlier_percentage']:.1f}%)")
    print(f"   • Severe outliers: {len(consensus_results['severe_outliers']):,}")
    print(f"   • Moderate outliers: {len(consensus_results['moderate_outliers']):,}")
    print(f"   • Mild outliers: {len(consensus_results['mild_outliers']):,}")
    
    print(f"\n🍷 DOMAIN-SPECIFIC FINDINGS:")
    if domain_specific:
        print(f"   • Chemically impossible values: {len(domain_specific.get('chemically_impossible', []))}")
        print(f"   • Extremely rare values: {len(domain_specific.get('extremely_rare', []))}")
        print(f"   • Suspicious combinations: {len(domain_specific.get('suspicious_combinations', []))}")
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    for category, recommendations in treatment_recommendations.items():
        if isinstance(recommendations, list) and recommendations:
            print(f"   {category.replace('_', ' ').title()}:")
            for rec in recommendations[:2]:  # Show first 2 recommendations
                print(f"     • {rec}")
    
    print(f"\n📁 Visualizations saved to: {output_dir}/")
    print(f"   • 6 comprehensive outlier analysis plots generated")
    
    print("\n" + "="*70)
    print("ADVANCED OUTLIER ANALYSIS COMPLETE")
    print("="*70)
    
    return final_results

# Example usage
if __name__ == "__main__":
    # Load wine dataset for testing
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Perform comprehensive outlier analysis
    analysis_results = comprehensive_outlier_analysis(df)