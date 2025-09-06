import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import networkx as nx
import warnings
import os
from typing import Dict, List, Tuple, Any
from itertools import combinations
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Comprehensive correlation and multivariate analysis toolkit.
    
    Provides Pearson and Spearman correlations, significance testing,
    multicollinearity detection, network visualization, and targeted
    analysis of relationships with wine sales (TARGET variable).
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'TARGET', 
                 output_dir: str = 'correlation_analysis_plots'):
        """
        Initialize the correlation analyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to analyze
        target_column : str, default 'TARGET'
            Name of the target variable column
        output_dir : str
            Directory to save visualization plots
        """
        self.df = df.copy()
        self.target_column = target_column
        self.output_dir = output_dir
        self.numerical_columns = self._identify_numerical_columns()
        self.results = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("RdBu_r")
    
    def _identify_numerical_columns(self) -> List[str]:
        """Identify numerical columns suitable for correlation analysis."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove index-like columns
        numerical_cols = [col for col in numerical_cols if not col.upper().startswith('INDEX')]
        return numerical_cols
    
    def calculate_correlation_matrices(self) -> Dict[str, Any]:
        """
        Calculate Pearson and Spearman correlation matrices with significance testing.
        
        Returns:
        --------
        dict
            Dictionary containing correlation matrices and p-value matrices
        """
        print("Calculating correlation matrices...")
        
        # Get clean numerical data
        clean_df = self.df[self.numerical_columns].dropna()
        
        # Calculate Pearson correlations
        pearson_corr = clean_df.corr(method='pearson')
        
        # Calculate Spearman correlations  
        spearman_corr = clean_df.corr(method='spearman')
        
        # Calculate p-values for Pearson correlations
        pearson_pvalues = pd.DataFrame(index=pearson_corr.index, columns=pearson_corr.columns)
        spearman_pvalues = pd.DataFrame(index=spearman_corr.index, columns=spearman_corr.columns)
        
        for col1 in self.numerical_columns:
            for col2 in self.numerical_columns:
                if col1 == col2:
                    pearson_pvalues.loc[col1, col2] = 0.0
                    spearman_pvalues.loc[col1, col2] = 0.0
                else:
                    # Get data for both variables, removing NaN pairs
                    data1 = self.df[col1].dropna()
                    data2 = self.df[col2].dropna()
                    
                    # Find common indices
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 2:
                        x = data1.loc[common_idx]
                        y = data2.loc[common_idx]
                        
                        # Calculate Pearson p-value
                        try:
                            _, p_pearson = pearsonr(x, y)
                            pearson_pvalues.loc[col1, col2] = p_pearson
                        except:
                            pearson_pvalues.loc[col1, col2] = np.nan
                        
                        # Calculate Spearman p-value
                        try:
                            _, p_spearman = spearmanr(x, y)
                            spearman_pvalues.loc[col1, col2] = p_spearman
                        except:
                            spearman_pvalues.loc[col1, col2] = np.nan
                    else:
                        pearson_pvalues.loc[col1, col2] = np.nan
                        spearman_pvalues.loc[col1, col2] = np.nan
        
        # Convert p-values to numeric
        pearson_pvalues = pearson_pvalues.astype(float)
        spearman_pvalues = spearman_pvalues.astype(float)
        
        correlation_results = {
            'pearson_correlations': pearson_corr,
            'spearman_correlations': spearman_corr,
            'pearson_pvalues': pearson_pvalues,
            'spearman_pvalues': spearman_pvalues,
            'sample_size': len(clean_df)
        }
        
        return correlation_results
    
    def create_correlation_heatmaps(self, correlation_results: Dict[str, Any]) -> None:
        """
        Create professional correlation heatmaps with significance indicators.
        
        Parameters:
        -----------
        correlation_results : dict
            Results from calculate_correlation_matrices()
        """
        print("Creating correlation heatmaps...")
        
        # Pearson correlation heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Pearson heatmap
        mask_pearson = correlation_results['pearson_pvalues'] > 0.05
        sns.heatmap(correlation_results['pearson_correlations'], 
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation Coefficient'},
                   mask=mask_pearson, annot_kws={'size': 8})
        ax1.set_title('Pearson Correlations\n(Only significant correlations shown, p < 0.05)', 
                     fontsize=14, fontweight='bold')
        
        # Spearman heatmap
        mask_spearman = correlation_results['spearman_pvalues'] > 0.05
        sns.heatmap(correlation_results['spearman_correlations'], 
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                   square=True, ax=ax2, cbar_kws={'label': 'Correlation Coefficient'},
                   mask=mask_spearman, annot_kws={'size': 8})
        ax2.set_title('Spearman Correlations\n(Only significant correlations shown, p < 0.05)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmaps_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed Pearson heatmap with p-value annotations
        self._create_detailed_correlation_heatmap(correlation_results)
    
    def _create_detailed_correlation_heatmap(self, correlation_results: Dict[str, Any]) -> None:
        """Create detailed correlation heatmap with significance stars."""
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create significance annotations
        corr_matrix = correlation_results['pearson_correlations']
        p_matrix = correlation_results['pearson_pvalues']
        
        # Create annotation matrix with significance stars
        annotations = corr_matrix.round(3).astype(str)
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                p_val = p_matrix.iloc[i, j]
                corr_val = corr_matrix.iloc[i, j]
                
                if pd.isna(p_val):
                    annotations.iloc[i, j] = 'N/A'
                elif i == j:
                    annotations.iloc[i, j] = '1.000'
                else:
                    if p_val < 0.001:
                        star = '***'
                    elif p_val < 0.01:
                        star = '**'
                    elif p_val < 0.05:
                        star = '*'
                    else:
                        star = ''
                    
                    annotations.iloc[i, j] = f'{corr_val:.3f}{star}'
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=annotations, fmt='', cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Pearson Correlation'},
                   annot_kws={'size': 7})
        
        ax.set_title('Detailed Pearson Correlations with Significance\n*** p<0.001, ** p<0.01, * p<0.05', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/detailed_pearson_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_vif_multicollinearity(self) -> Dict[str, Any]:
        """
        Perform Variance Inflation Factor analysis for multicollinearity detection.
        
        Returns:
        --------
        dict
            VIF analysis results
        """
        print("Analyzing multicollinearity using VIF...")
        
        # Get clean numerical data
        clean_df = self.df[self.numerical_columns].dropna()
        
        if len(clean_df) < 10:
            return {'error': 'Insufficient data for VIF analysis'}
        
        vif_results = []
        
        try:
            # Calculate VIF for each variable
            for i, column in enumerate(self.numerical_columns):
                # Skip if column has no variance
                if clean_df[column].var() == 0:
                    vif_results.append({
                        'Variable': column,
                        'VIF': np.inf,
                        'Interpretation': 'No variance - constant variable'
                    })
                    continue
                
                try:
                    vif_value = variance_inflation_factor(clean_df.values, i)
                    
                    # Interpret VIF value
                    if vif_value < 1:
                        interpretation = 'No correlation'
                    elif vif_value < 5:
                        interpretation = 'Low multicollinearity'
                    elif vif_value < 10:
                        interpretation = 'Moderate multicollinearity'
                    elif vif_value < 100:
                        interpretation = 'High multicollinearity'
                    else:
                        interpretation = 'Extreme multicollinearity'
                    
                    vif_results.append({
                        'Variable': column,
                        'VIF': vif_value,
                        'Interpretation': interpretation
                    })
                    
                except Exception as e:
                    vif_results.append({
                        'Variable': column,
                        'VIF': np.nan,
                        'Interpretation': f'Calculation error: {str(e)[:50]}'
                    })
        
        except Exception as e:
            return {'error': f'VIF analysis failed: {str(e)}'}
        
        vif_df = pd.DataFrame(vif_results)
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        # Create VIF visualization
        self._create_vif_visualization(vif_df)
        
        return {
            'vif_results': vif_df,
            'high_vif_variables': vif_df[vif_df['VIF'] > 10]['Variable'].tolist(),
            'moderate_vif_variables': vif_df[(vif_df['VIF'] >= 5) & (vif_df['VIF'] <= 10)]['Variable'].tolist()
        }
    
    def _create_vif_visualization(self, vif_df: pd.DataFrame) -> None:
        """Create VIF visualization."""
        
        plt.figure(figsize=(12, 8))
        
        # Filter out infinite and NaN values for plotting
        plot_data = vif_df[vif_df['VIF'].notna() & (vif_df['VIF'] != np.inf)].copy()
        
        if len(plot_data) == 0:
            plt.text(0.5, 0.5, 'No valid VIF values to plot', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=16)
            plt.title('VIF Analysis - No Valid Data')
        else:
            # Create color map based on VIF values
            colors = []
            for vif in plot_data['VIF']:
                if vif < 5:
                    colors.append('green')
                elif vif < 10:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            bars = plt.barh(range(len(plot_data)), plot_data['VIF'], color=colors, alpha=0.7)
            plt.yticks(range(len(plot_data)), plot_data['Variable'])
            plt.xlabel('Variance Inflation Factor (VIF)')
            plt.title('Multicollinearity Analysis - VIF Values', fontsize=16, fontweight='bold')
            
            # Add reference lines
            plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold (VIF=5)')
            plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='High threshold (VIF=10)')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if not np.isnan(width) and width != np.inf:
                    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                            f'{width:.2f}', ha='left', va='center')
            
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/vif_multicollinearity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_target_correlations(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations specifically with the TARGET variable.
        
        Parameters:
        -----------
        correlation_results : dict
            Results from calculate_correlation_matrices()
            
        Returns:
        --------
        dict
            Detailed TARGET correlation analysis
        """
        print(f"Analyzing correlations with {self.target_column} variable...")
        
        if self.target_column not in self.numerical_columns:
            return {'error': f'{self.target_column} not found in numerical columns'}
        
        # Extract correlations with TARGET
        pearson_target = correlation_results['pearson_correlations'][self.target_column].drop(self.target_column)
        spearman_target = correlation_results['spearman_correlations'][self.target_column].drop(self.target_column)
        pearson_p_target = correlation_results['pearson_pvalues'][self.target_column].drop(self.target_column)
        spearman_p_target = correlation_results['spearman_pvalues'][self.target_column].drop(self.target_column)
        
        # Create TARGET correlation DataFrame
        target_corr_df = pd.DataFrame({
            'Variable': pearson_target.index,
            'Pearson_Correlation': pearson_target.values,
            'Pearson_P_Value': pearson_p_target.values,
            'Spearman_Correlation': spearman_target.values,
            'Spearman_P_Value': spearman_p_target.values
        })
        
        # Add significance indicators
        target_corr_df['Pearson_Significant'] = target_corr_df['Pearson_P_Value'] < 0.05
        target_corr_df['Spearman_Significant'] = target_corr_df['Spearman_P_Value'] < 0.05
        
        # Sort by absolute Pearson correlation
        target_corr_df['Abs_Pearson'] = abs(target_corr_df['Pearson_Correlation'])
        target_corr_df = target_corr_df.sort_values('Abs_Pearson', ascending=False)
        
        # Identify strongly correlated variables
        strong_positive = target_corr_df[
            (target_corr_df['Pearson_Correlation'] > 0.3) & 
            (target_corr_df['Pearson_Significant'])
        ].copy()
        
        strong_negative = target_corr_df[
            (target_corr_df['Pearson_Correlation'] < -0.3) & 
            (target_corr_df['Pearson_Significant'])
        ].copy()
        
        # Create TARGET correlation visualization
        self._create_target_correlation_plots(target_corr_df)
        
        return {
            'target_correlations': target_corr_df,
            'strong_positive_correlations': strong_positive,
            'strong_negative_correlations': strong_negative,
            'top_5_positive': target_corr_df[target_corr_df['Pearson_Correlation'] > 0].head(5),
            'top_5_negative': target_corr_df[target_corr_df['Pearson_Correlation'] < 0].head(5)
        }
    
    def _create_target_correlation_plots(self, target_corr_df: pd.DataFrame) -> None:
        """Create TARGET correlation visualization plots."""
        
        # Plot 1: Bar chart of correlations with TARGET
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Pearson correlations
        colors = ['red' if x < 0 else 'blue' for x in target_corr_df['Pearson_Correlation']]
        bars1 = ax1.barh(range(len(target_corr_df)), target_corr_df['Pearson_Correlation'], 
                        color=colors, alpha=0.7)
        ax1.set_yticks(range(len(target_corr_df)))
        ax1.set_yticklabels(target_corr_df['Variable'])
        ax1.set_xlabel('Pearson Correlation with TARGET')
        ax1.set_title('Pearson Correlations with Wine Sales (TARGET)', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add significance indicators
        for i, (idx, row) in enumerate(target_corr_df.iterrows()):
            if row['Pearson_Significant']:
                ax1.text(row['Pearson_Correlation'] + (0.02 if row['Pearson_Correlation'] > 0 else -0.02), 
                        i, '*', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Spearman correlations
        colors = ['red' if x < 0 else 'blue' for x in target_corr_df['Spearman_Correlation']]
        bars2 = ax2.barh(range(len(target_corr_df)), target_corr_df['Spearman_Correlation'], 
                        color=colors, alpha=0.7)
        ax2.set_yticks(range(len(target_corr_df)))
        ax2.set_yticklabels(target_corr_df['Variable'])
        ax2.set_xlabel('Spearman Correlation with TARGET')
        ax2.set_title('Spearman Correlations with Wine Sales (TARGET)', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add significance indicators
        for i, (idx, row) in enumerate(target_corr_df.iterrows()):
            if row['Spearman_Significant']:
                ax2.text(row['Spearman_Correlation'] + (0.02 if row['Spearman_Correlation'] > 0 else -0.02), 
                        i, '*', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add legend
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative correlation')
        blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Positive correlation')
        ax1.legend(handles=[red_patch, blue_patch], loc='lower right')
        ax2.legend(handles=[red_patch, blue_patch], loc='lower right')
        
        plt.figtext(0.5, 0.02, '* indicates statistical significance (p < 0.05)', 
                   ha='center', fontsize=10, style='italic')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        plt.savefig(f'{self.output_dir}/target_correlations_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def identify_highly_correlated_pairs(self, correlation_results: Dict[str, Any], 
                                       threshold: float = 0.7) -> Dict[str, Any]:
        """
        Identify highly correlated variable pairs.
        
        Parameters:
        -----------
        correlation_results : dict
            Results from calculate_correlation_matrices()
        threshold : float, default 0.7
            Absolute correlation threshold for identification
            
        Returns:
        --------
        dict
            Highly correlated pairs analysis
        """
        print(f"Identifying highly correlated pairs (threshold = {threshold})...")
        
        corr_matrix = correlation_results['pearson_correlations']
        p_matrix = correlation_results['pearson_pvalues']
        
        high_corr_pairs = []
        
        # Find pairs with high correlation
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                p_value = p_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold and p_value < 0.05:
                    high_corr_pairs.append({
                        'Variable_1': var1,
                        'Variable_2': var2,
                        'Correlation': correlation,
                        'Abs_Correlation': abs(correlation),
                        'P_Value': p_value,
                        'Relationship': 'Strong Positive' if correlation > 0 else 'Strong Negative'
                    })
        
        high_corr_df = pd.DataFrame(high_corr_pairs)
        
        if not high_corr_df.empty:
            high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
        
        return {
            'highly_correlated_pairs': high_corr_df,
            'total_pairs': len(high_corr_pairs),
            'threshold_used': threshold
        }
    
    def create_scatter_plot_matrix(self, max_variables: int = 8) -> None:
        """
        Create scatter plot matrix for key variable relationships.
        
        Parameters:
        -----------
        max_variables : int, default 8
            Maximum number of variables to include in scatter plot matrix
        """
        print(f"Creating scatter plot matrix (max {max_variables} variables)...")
        
        # Select key variables (including TARGET if available)
        key_variables = self.numerical_columns[:max_variables]
        if self.target_column in self.numerical_columns and self.target_column not in key_variables:
            key_variables = [self.target_column] + key_variables[:max_variables-1]
        
        # Get clean data for selected variables
        plot_data = self.df[key_variables].dropna()
        
        if len(plot_data) < 10:
            print("Insufficient data for scatter plot matrix")
            return
        
        # Create scatter plot matrix
        fig = plt.figure(figsize=(16, 16))
        
        # Use seaborn's pairplot for better visualization
        g = sns.PairGrid(plot_data, height=2.5)
        g.map_upper(sns.scatterplot, alpha=0.6, s=20)
        g.map_lower(sns.scatterplot, alpha=0.6, s=20)
        g.map_diag(sns.histplot, kde=True, alpha=0.7)
        
        # Add correlation coefficients to upper triangle
        def add_corr(x, y, **kwargs):
            r = stats.pearsonr(x, y)[0]
            ax = plt.gca()
            ax.annotate(f'r = {r:.3f}', xy=(.5, .5), xycoords=ax.transAxes,
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        g.map_upper(add_corr)
        
        plt.suptitle('Scatter Plot Matrix - Key Variable Relationships', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/scatter_plot_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_network(self, correlation_results: Dict[str, Any], 
                                 threshold: float = 0.5) -> None:
        """
        Create network visualization of correlations above threshold.
        
        Parameters:
        -----------
        correlation_results : dict
            Results from calculate_correlation_matrices()
        threshold : float, default 0.5
            Minimum absolute correlation for network connections
        """
        print(f"Creating correlation network (threshold = {threshold})...")
        
        corr_matrix = correlation_results['pearson_correlations']
        p_matrix = correlation_results['pearson_pvalues']
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for variable in corr_matrix.columns:
            G.add_node(variable)
        
        # Add edges for significant correlations above threshold
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                p_value = p_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold and p_value < 0.05:
                    G.add_edge(var1, var2, weight=abs(correlation), 
                              correlation=correlation, p_value=p_value)
        
        # Create network visualization
        plt.figure(figsize=(14, 10))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges with thickness proportional to correlation strength
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] * 3 for u, v in edges]  # Scale for visibility
        edge_colors = ['red' if G[u][v]['correlation'] < 0 else 'blue' for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color=edge_colors)
        
        # Draw nodes
        node_colors = ['gold' if node == self.target_column else 'lightblue' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Correlation Network (|r| ≥ {threshold}, p < 0.05)\n' + 
                 f'Red = Negative correlation, Blue = Positive correlation', 
                 fontsize=16, fontweight='bold')
        
        # Add legend
        red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Negative correlation')
        blue_line = plt.Line2D([0], [0], color='blue', linewidth=3, label='Positive correlation')
        target_patch = mpatches.Patch(color='gold', label=f'{self.target_column} (Target)')
        var_patch = mpatches.Patch(color='lightblue', label='Other variables')
        
        plt.legend(handles=[red_line, blue_line, target_patch, var_patch], 
                  loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_correlation_report(self, correlation_results: Dict[str, Any],
                                  vif_results: Dict[str, Any],
                                  target_analysis: Dict[str, Any],
                                  high_corr_pairs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive correlation analysis report.
        
        Returns:
        --------
        dict
            Comprehensive correlation analysis report
        """
        print("Generating correlation analysis report...")
        
        report = {
            'dataset_summary': {
                'total_variables': len(self.numerical_columns),
                'sample_size': correlation_results['sample_size'],
                'target_variable': self.target_column
            },
            'correlation_summary': {
                'pearson_correlations': correlation_results['pearson_correlations'],
                'spearman_correlations': correlation_results['spearman_correlations'],
                'significant_correlations_count': (correlation_results['pearson_pvalues'] < 0.05).sum().sum() // 2
            },
            'multicollinearity_analysis': vif_results,
            'target_analysis': target_analysis,
            'highly_correlated_pairs': high_corr_pairs,
            'key_insights': self._generate_key_insights(correlation_results, target_analysis, high_corr_pairs),
            'recommendations': self._generate_recommendations(vif_results, target_analysis, high_corr_pairs)
        }
        
        return report
    
    def _generate_key_insights(self, correlation_results: Dict[str, Any],
                              target_analysis: Dict[str, Any],
                              high_corr_pairs: Dict[str, Any]) -> List[str]:
        """Generate key insights from correlation analysis."""
        
        insights = []
        
        # Target variable insights
        if 'target_correlations' in target_analysis:
            target_corr = target_analysis['target_correlations']
            strongest_positive = target_corr.iloc[0] if target_corr.iloc[0]['Pearson_Correlation'] > 0 else None
            strongest_negative = target_corr.iloc[0] if target_corr.iloc[0]['Pearson_Correlation'] < 0 else None
            
            if strongest_positive is not None:
                insights.append(f"Strongest positive predictor of wine sales: {strongest_positive['Variable']} (r = {strongest_positive['Pearson_Correlation']:.3f})")
            
            if strongest_negative is not None:
                insights.append(f"Strongest negative predictor of wine sales: {strongest_negative['Variable']} (r = {strongest_negative['Pearson_Correlation']:.3f})")
            
            # Count significant correlations with target
            sig_count = (target_corr['Pearson_Significant']).sum()
            insights.append(f"Number of variables significantly correlated with wine sales: {sig_count}")
        
        # Multicollinearity insights
        if high_corr_pairs['total_pairs'] > 0:
            insights.append(f"Found {high_corr_pairs['total_pairs']} highly correlated variable pairs (|r| ≥ 0.7)")
        
        return insights
    
    def _generate_recommendations(self, vif_results: Dict[str, Any],
                                target_analysis: Dict[str, Any],
                                high_corr_pairs: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # VIF recommendations
        if 'high_vif_variables' in vif_results and len(vif_results['high_vif_variables']) > 0:
            recommendations.append(f"Consider removing variables with high VIF (>10): {vif_results['high_vif_variables']}")
        
        # Target correlation recommendations
        if 'strong_positive_correlations' in target_analysis and not target_analysis['strong_positive_correlations'].empty:
            strong_vars = target_analysis['strong_positive_correlations']['Variable'].tolist()
            recommendations.append(f"Focus on positive predictors for wine sales improvement: {strong_vars}")
        
        # Multicollinearity recommendations
        if high_corr_pairs['total_pairs'] > 5:
            recommendations.append("High multicollinearity detected - consider dimensionality reduction techniques")
        
        # General recommendations
        recommendations.append("Use correlation results to inform feature selection for predictive modeling")
        recommendations.append("Consider both Pearson and Spearman correlations for comprehensive analysis")
        
        return recommendations

def comprehensive_correlation_analysis(df: pd.DataFrame, target_column: str = 'TARGET',
                                     output_dir: str = 'correlation_analysis_plots') -> Dict[str, Any]:
    """
    Perform comprehensive correlation and multivariate analysis.
    
    This function provides extensive correlation analysis including:
    - Pearson and Spearman correlation matrices with significance testing
    - Professional correlation heatmaps with significance indicators
    - VIF analysis for multicollinearity detection
    - Scatter plot matrix for key variable relationships
    - Targeted analysis of correlations with wine sales (TARGET)
    - Correlation network visualization
    - Identification of highly correlated pairs
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    target_column : str, default 'TARGET'
        Name of the target variable (wine sales)
    output_dir : str, default 'correlation_analysis_plots'
        Directory to save visualization plots
        
    Returns:
    --------
    dict
        Comprehensive correlation analysis results including:
        - Correlation matrices (Pearson & Spearman)
        - Statistical significance testing results
        - VIF multicollinearity analysis
        - TARGET variable correlation analysis
        - Highly correlated pairs identification
        - Professional visualizations saved to files
        
    Generated Visualizations:
    -------------------------
    - correlation_heatmaps_combined.png: Side-by-side Pearson & Spearman heatmaps
    - detailed_pearson_correlations.png: Detailed heatmap with significance stars
    - vif_multicollinearity_analysis.png: VIF bar chart for multicollinearity
    - target_correlations_analysis.png: Correlations with wine sales (TARGET)
    - scatter_plot_matrix.png: Pairwise scatter plots for key variables
    - correlation_network.png: Network visualization of strong correlations
    """
    
    print("="*70)
    print("COMPREHENSIVE CORRELATION & MULTIVARIATE ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer(df, target_column, output_dir)
    
    # Run correlation analysis
    correlation_results = analyzer.calculate_correlation_matrices()
    analyzer.create_correlation_heatmaps(correlation_results)
    
    # VIF multicollinearity analysis
    vif_results = analyzer.analyze_vif_multicollinearity()
    
    # TARGET variable analysis
    target_analysis = analyzer.analyze_target_correlations(correlation_results)
    
    # Highly correlated pairs
    high_corr_pairs = analyzer.identify_highly_correlated_pairs(correlation_results, threshold=0.7)
    
    # Create visualizations
    analyzer.create_scatter_plot_matrix(max_variables=8)
    analyzer.create_correlation_network(correlation_results, threshold=0.5)
    
    # Generate comprehensive report
    final_report = analyzer.generate_correlation_report(
        correlation_results, vif_results, target_analysis, high_corr_pairs)
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   • Variables analyzed: {len(analyzer.numerical_columns)}")
    print(f"   • Sample size: {correlation_results['sample_size']:,}")
    print(f"   • Significant correlations: {(correlation_results['pearson_pvalues'] < 0.05).sum().sum() // 2}")
    print(f"   • Highly correlated pairs (|r| ≥ 0.7): {high_corr_pairs['total_pairs']}")
    
    if 'high_vif_variables' in vif_results:
        print(f"   • High multicollinearity variables (VIF > 10): {len(vif_results['high_vif_variables'])}")
    
    print(f"\n🎯 TARGET VARIABLE INSIGHTS:")
    if 'target_correlations' in target_analysis:
        target_corr = target_analysis['target_correlations']
        top_positive = target_corr[target_corr['Pearson_Correlation'] > 0].iloc[0] if any(target_corr['Pearson_Correlation'] > 0) else None
        top_negative = target_corr[target_corr['Pearson_Correlation'] < 0].iloc[0] if any(target_corr['Pearson_Correlation'] < 0) else None
        
        if top_positive is not None:
            print(f"   • Strongest positive correlation: {top_positive['Variable']} (r = {top_positive['Pearson_Correlation']:.3f})")
        if top_negative is not None:
            print(f"   • Strongest negative correlation: {top_negative['Variable']} (r = {top_negative['Pearson_Correlation']:.3f})")
    
    print(f"\n💡 KEY INSIGHTS:")
    for insight in final_report['key_insights']:
        print(f"   • {insight}")
    
    print(f"\n📁 Visualizations saved to: {output_dir}/")
    print(f"   • 6 comprehensive correlation analysis plots generated")
    
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS COMPLETE")
    print("="*70)
    
    return final_report

# Example usage
if __name__ == "__main__":
    # Load wine dataset for testing
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Perform comprehensive correlation analysis
    analysis_results = comprehensive_correlation_analysis(df, target_column='TARGET')