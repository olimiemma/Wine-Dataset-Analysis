import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
import os
from typing import Dict, List, Tuple, Any, Union
warnings.filterwarnings('ignore')

class TargetAnalyzer:
    """
    Specialized analysis system for wine sales (TARGET) variable.
    
    Provides comprehensive distribution analysis, relationship exploration,
    predictive feature identification, and business insights for wine sales.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'TARGET', 
                 output_dir: str = 'target_analysis_plots'):
        """
        Initialize the TARGET analyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Wine dataset
        target_column : str, default 'TARGET'
            Name of the target variable (wine sales)
        output_dir : str
            Directory to save visualization plots
        """
        self.df = df.copy()
        self.target_column = target_column
        self.output_dir = output_dir
        self.numerical_columns = self._identify_numerical_columns()
        self.chemical_columns = self._identify_chemical_columns()
        self.quality_columns = self._identify_quality_columns()
        self.results = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("viridis")
    
    def _identify_numerical_columns(self) -> List[str]:
        """Identify numerical columns excluding the target."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target and index-like columns
        numerical_cols = [col for col in numerical_cols 
                         if col != self.target_column and not col.upper().startswith('INDEX')]
        return numerical_cols
    
    def _identify_chemical_columns(self) -> List[str]:
        """Identify wine chemistry-related columns."""
        chemical_keywords = ['Acid', 'pH', 'Alcohol', 'Sugar', 'Chloride', 'Sulfur', 'Sulphate', 'Density']
        chemical_cols = []
        
        for col in self.numerical_columns:
            if any(keyword in col for keyword in chemical_keywords):
                chemical_cols.append(col)
        
        return chemical_cols
    
    def _identify_quality_columns(self) -> List[str]:
        """Identify wine quality-related columns."""
        quality_keywords = ['STARS', 'Appeal', 'Label']
        quality_cols = []
        
        for col in self.numerical_columns:
            if any(keyword in col for keyword in quality_keywords):
                quality_cols.append(col)
        
        return quality_cols
    
    def analyze_target_distribution(self) -> Dict[str, Any]:
        """
        Comprehensive TARGET variable distribution analysis.
        
        Returns:
        --------
        dict
            Complete distribution analysis results
        """
        print(f"Analyzing {self.target_column} distribution...")
        
        target_data = self.df[self.target_column].dropna()
        
        # Basic statistics
        distribution_stats = {
            'count': len(target_data),
            'missing_count': self.df[self.target_column].isnull().sum(),
            'missing_percentage': (self.df[self.target_column].isnull().sum() / len(self.df)) * 100,
            'unique_values': target_data.nunique(),
            'value_counts': target_data.value_counts().sort_index(),
            'mean': target_data.mean(),
            'median': target_data.median(),
            'mode': target_data.mode().iloc[0] if not target_data.mode().empty else np.nan,
            'std': target_data.std(),
            'variance': target_data.var(),
            'min': target_data.min(),
            'max': target_data.max(),
            'range': target_data.max() - target_data.min(),
            'q1': target_data.quantile(0.25),
            'q3': target_data.quantile(0.75),
            'iqr': target_data.quantile(0.75) - target_data.quantile(0.25),
            'skewness': target_data.skew(),
            'kurtosis': target_data.kurtosis(),
            'cv': (target_data.std() / target_data.mean()) * 100 if target_data.mean() != 0 else np.inf
        }
        
        # Distribution shape analysis
        distribution_stats['skewness_interpretation'] = self._interpret_skewness(distribution_stats['skewness'])
        distribution_stats['kurtosis_interpretation'] = self._interpret_kurtosis(distribution_stats['kurtosis'])
        
        # Percentile analysis
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            distribution_stats[f'p{p}'] = target_data.quantile(p/100)
        
        # Business interpretation
        distribution_stats['business_insights'] = self._generate_distribution_insights(distribution_stats)
        
        # Create distribution visualizations
        self._create_distribution_plots(target_data, distribution_stats)
        
        return distribution_stats
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness values for business context."""
        if abs(skewness) < 0.5:
            return "Symmetric distribution - balanced sales across range"
        elif -1 < skewness < -0.5:
            return "Left-skewed - more wines with higher sales"
        elif 0.5 < skewness < 1:
            return "Right-skewed - more wines with lower sales"
        elif skewness <= -1:
            return "Highly left-skewed - most wines are high performers"
        elif skewness >= 1:
            return "Highly right-skewed - most wines are low performers"
        else:
            return "Symmetric"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis values for business context."""
        if abs(kurtosis) < 0.5:
            return "Normal tail thickness - typical sales distribution"
        elif kurtosis > 0.5:
            return "Heavy tails - more extreme sales values than normal"
        else:
            return "Light tails - fewer extreme sales values"
    
    def _generate_distribution_insights(self, stats: Dict[str, Any]) -> List[str]:
        """Generate business insights from distribution statistics."""
        insights = []
        
        # Sales range analysis
        sales_range = stats['max'] - stats['min']
        insights.append(f"Wine sales range from {stats['min']} to {stats['max']} (range: {sales_range})")
        
        # Central tendency
        if abs(stats['mean'] - stats['median']) / stats['mean'] > 0.1:
            if stats['mean'] > stats['median']:
                insights.append("Mean > Median: A few high-selling wines pull up the average")
            else:
                insights.append("Mean < Median: Distribution concentrated in higher sales range")
        else:
            insights.append("Mean ≈ Median: Balanced sales distribution")
        
        # Variability
        if stats['cv'] > 50:
            insights.append("High sales variability - wide range in wine performance")
        elif stats['cv'] > 25:
            insights.append("Moderate sales variability - some performance differences")
        else:
            insights.append("Low sales variability - consistent wine performance")
        
        # Market segments
        q1, median, q3 = stats['q1'], stats['median'], stats['q3']
        insights.append(f"Bottom 25% wines: sales ≤ {q1}")
        insights.append(f"Top 25% wines: sales ≥ {q3}")
        insights.append(f"Middle 50% wines: sales between {q1} and {q3}")
        
        return insights
    
    def analyze_zero_inflation(self) -> Dict[str, Any]:
        """
        Analyze zero-inflation in wine sales data.
        
        Returns:
        --------
        dict
            Zero-inflation analysis results
        """
        print("Analyzing zero-inflation in wine sales...")
        
        target_data = self.df[self.target_column].dropna()
        
        # Count zeros and near-zeros
        zero_count = (target_data == 0).sum()
        near_zero_count = (target_data <= 0.1).sum()  # Adjust threshold as needed
        
        # Calculate proportions
        zero_proportion = zero_count / len(target_data)
        near_zero_proportion = near_zero_count / len(target_data)
        
        # Expected zeros under Poisson distribution (if applicable)
        if target_data.min() >= 0 and all(target_data == target_data.astype(int)):
            lambda_param = target_data.mean()
            expected_zeros_poisson = stats.poisson.pmf(0, lambda_param) * len(target_data)
            excess_zeros = zero_count - expected_zeros_poisson
        else:
            expected_zeros_poisson = np.nan
            excess_zeros = np.nan
        
        # Business interpretation
        zero_inflation_level = "None"
        business_impact = []
        
        if zero_proportion > 0.2:
            zero_inflation_level = "High"
            business_impact.append("High proportion of non-selling wines - investigate market fit")
        elif zero_proportion > 0.1:
            zero_inflation_level = "Moderate" 
            business_impact.append("Moderate zero-sales - some wines may need repositioning")
        elif zero_proportion > 0.05:
            zero_inflation_level = "Low"
            business_impact.append("Low zero-sales rate - good overall market performance")
        else:
            zero_inflation_level = "Minimal"
            business_impact.append("Very few zero-sales wines - strong product lineup")
        
        # Value distribution analysis
        value_distribution = {
            'zero_sales': zero_count,
            'low_sales_1_2': ((target_data > 0) & (target_data <= 2)).sum(),
            'moderate_sales_3_5': ((target_data > 2) & (target_data <= 5)).sum(), 
            'good_sales_6_8': ((target_data > 5) & (target_data <= 8)).sum(),
            'excellent_sales_above_8': (target_data > 8).sum()
        }
        
        zero_analysis = {
            'zero_count': zero_count,
            'zero_proportion': zero_proportion,
            'near_zero_count': near_zero_count,
            'near_zero_proportion': near_zero_proportion,
            'expected_zeros_poisson': expected_zeros_poisson,
            'excess_zeros': excess_zeros,
            'zero_inflation_level': zero_inflation_level,
            'business_impact': business_impact,
            'value_distribution': value_distribution,
            'total_wines': len(target_data)
        }
        
        # Create zero-inflation visualization
        self._create_zero_inflation_plot(zero_analysis, target_data)
        
        return zero_analysis
    
    def analyze_target_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships between TARGET and all other variables.
        
        Returns:
        --------
        dict
            Comprehensive relationship analysis
        """
        print("Analyzing TARGET relationships with all variables...")
        
        relationships = {
            'chemical_relationships': {},
            'quality_relationships': {},
            'all_relationships': {},
            'correlation_summary': {},
            'top_positive_predictors': [],
            'top_negative_predictors': []
        }
        
        target_data = self.df[self.target_column].dropna()
        
        # Analyze relationships with each variable
        all_correlations = []
        
        for column in self.numerical_columns:
            if column in self.df.columns:
                # Get paired data (remove rows where either variable is missing)
                paired_data = self.df[[self.target_column, column]].dropna()
                
                if len(paired_data) > 10:  # Minimum sample size
                    target_values = paired_data[self.target_column]
                    predictor_values = paired_data[column]
                    
                    # Calculate correlations
                    pearson_r, pearson_p = stats.pearsonr(target_values, predictor_values)
                    spearman_r, spearman_p = stats.spearmanr(target_values, predictor_values)
                    
                    # Calculate mutual information (non-linear relationships)
                    try:
                        mutual_info = mutual_info_regression(
                            predictor_values.values.reshape(-1, 1), 
                            target_values.values, 
                            random_state=42
                        )[0]
                    except:
                        mutual_info = np.nan
                    
                    relationship_data = {
                        'variable': column,
                        'pearson_correlation': pearson_r,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_r,
                        'spearman_p_value': spearman_p,
                        'mutual_information': mutual_info,
                        'sample_size': len(paired_data),
                        'relationship_strength': self._categorize_correlation(abs(pearson_r)),
                        'business_interpretation': self._interpret_business_relationship(column, pearson_r, pearson_p)
                    }
                    
                    all_correlations.append(relationship_data)
                    relationships['all_relationships'][column] = relationship_data
                    
                    # Categorize by variable type
                    if column in self.chemical_columns:
                        relationships['chemical_relationships'][column] = relationship_data
                    elif column in self.quality_columns:
                        relationships['quality_relationships'][column] = relationship_data
        
        # Sort by absolute correlation strength
        all_correlations.sort(key=lambda x: abs(x['pearson_correlation']), reverse=True)
        
        # Identify top predictors
        significant_correlations = [rel for rel in all_correlations if rel['pearson_p_value'] < 0.05]
        
        positive_predictors = [rel for rel in significant_correlations if rel['pearson_correlation'] > 0]
        negative_predictors = [rel for rel in significant_correlations if rel['pearson_correlation'] < 0]
        
        relationships['top_positive_predictors'] = positive_predictors[:5]
        relationships['top_negative_predictors'] = negative_predictors[:5]
        relationships['all_correlations_ranked'] = all_correlations
        
        # Summary statistics
        significant_count = len(significant_correlations)
        strong_relationships = len([rel for rel in all_correlations if abs(rel['pearson_correlation']) > 0.3])
        
        relationships['correlation_summary'] = {
            'total_variables_analyzed': len(all_correlations),
            'significant_relationships': significant_count,
            'strong_relationships': strong_relationships,
            'strongest_positive': positive_predictors[0] if positive_predictors else None,
            'strongest_negative': negative_predictors[0] if negative_predictors else None
        }
        
        # Create relationship visualizations
        self._create_relationship_plots(relationships, target_data)
        
        return relationships
    
    def _categorize_correlation(self, abs_correlation: float) -> str:
        """Categorize correlation strength."""
        if abs_correlation >= 0.7:
            return "Very Strong"
        elif abs_correlation >= 0.5:
            return "Strong"
        elif abs_correlation >= 0.3:
            return "Moderate"
        elif abs_correlation >= 0.1:
            return "Weak"
        else:
            return "Very Weak"
    
    def _interpret_business_relationship(self, variable: str, correlation: float, p_value: float) -> str:
        """Generate business interpretation of variable relationships."""
        if p_value >= 0.05:
            return f"No significant relationship with sales"
        
        strength = "strongly" if abs(correlation) > 0.3 else "moderately" if abs(correlation) > 0.1 else "weakly"
        direction = "positively" if correlation > 0 else "negatively"
        
        # Variable-specific interpretations
        variable_interpretations = {
            'STARS': f"Wine ratings {strength} {direction} predict sales - {'higher ratings drive sales' if correlation > 0 else 'surprising negative relationship'}",
            'LabelAppeal': f"Label appeal {strength} {direction} impacts sales - {'marketing matters' if correlation > 0 else 'unexpected negative impact'}",
            'Alcohol': f"Alcohol content {strength} {direction} affects sales - {'consumers prefer stronger wines' if correlation > 0 else 'consumers prefer lighter wines'}",
            'pH': f"Wine pH {strength} {direction} influences sales - {'less acidic wines sell better' if correlation > 0 else 'more acidic wines sell better'}",
            'ResidualSugar': f"Sugar content {strength} {direction} correlates with sales - {'sweeter wines preferred' if correlation > 0 else 'drier wines preferred'}",
            'VolatileAcidity': f"Volatile acidity {strength} {direction} impacts sales - {'higher acidity reduces sales' if correlation < 0 else 'unexpected positive relationship'}"
        }
        
        return variable_interpretations.get(variable, f"{variable} {strength} {direction} correlates with sales")
    
    def identify_predictive_features(self) -> Dict[str, Any]:
        """
        Identify top predictive features using multiple methods.
        
        Returns:
        --------
        dict
            Predictive feature analysis results
        """
        print("Identifying top predictive features for wine sales...")
        
        # Prepare clean dataset
        feature_columns = self.numerical_columns
        analysis_df = self.df[feature_columns + [self.target_column]].dropna()
        
        if len(analysis_df) < 50:
            return {'error': 'Insufficient clean data for feature analysis'}
        
        X = analysis_df[feature_columns]
        y = analysis_df[self.target_column]
        
        predictive_analysis = {}
        
        # Method 1: Random Forest Feature Importance
        try:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            rf_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            predictive_analysis['random_forest'] = {
                'feature_importance': rf_importance,
                'model_score': rf_model.score(X, y),
                'top_5_features': rf_importance.head(5)['feature'].tolist()
            }
        except Exception as e:
            predictive_analysis['random_forest'] = {'error': str(e)}
        
        # Method 2: Mutual Information
        try:
            mutual_info_scores = mutual_info_regression(X, y, random_state=42)
            
            mutual_info_df = pd.DataFrame({
                'feature': feature_columns,
                'mutual_info_score': mutual_info_scores
            }).sort_values('mutual_info_score', ascending=False)
            
            predictive_analysis['mutual_information'] = {
                'feature_scores': mutual_info_df,
                'top_5_features': mutual_info_df.head(5)['feature'].tolist()
            }
        except Exception as e:
            predictive_analysis['mutual_information'] = {'error': str(e)}
        
        # Method 3: Correlation-based selection
        correlations = []
        for feature in feature_columns:
            if feature in analysis_df.columns:
                corr, p_val = stats.pearsonr(analysis_df[feature], analysis_df[self.target_column])
                correlations.append({
                    'feature': feature,
                    'abs_correlation': abs(corr),
                    'correlation': corr,
                    'p_value': p_val
                })
        
        correlation_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        significant_correlations = correlation_df[correlation_df['p_value'] < 0.05]
        
        predictive_analysis['correlation_based'] = {
            'feature_correlations': correlation_df,
            'significant_features': significant_correlations,
            'top_5_features': significant_correlations.head(5)['feature'].tolist()
        }
        
        # Create consensus ranking
        consensus_features = self._create_consensus_ranking(predictive_analysis)
        predictive_analysis['consensus_ranking'] = consensus_features
        
        # Create predictive features visualization
        self._create_predictive_features_plot(predictive_analysis)
        
        return predictive_analysis
    
    def _create_consensus_ranking(self, predictive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create consensus ranking of predictive features."""
        
        feature_scores = {}
        
        # Collect scores from each method
        methods = ['random_forest', 'mutual_information', 'correlation_based']
        
        for method in methods:
            if method in predictive_analysis and 'error' not in predictive_analysis[method]:
                top_features = predictive_analysis[method]['top_5_features']
                
                # Assign scores (5 for rank 1, 4 for rank 2, etc.)
                for rank, feature in enumerate(top_features):
                    score = 5 - rank
                    if feature not in feature_scores:
                        feature_scores[feature] = {'total_score': 0, 'methods': []}
                    feature_scores[feature]['total_score'] += score
                    feature_scores[feature]['methods'].append(f"{method}(rank_{rank+1})")
        
        # Sort by consensus score
        consensus_ranking = sorted(feature_scores.items(), 
                                 key=lambda x: x[1]['total_score'], reverse=True)
        
        consensus_features = {
            'ranking': consensus_ranking,
            'top_5_consensus': [item[0] for item in consensus_ranking[:5]],
            'scoring_details': feature_scores
        }
        
        return consensus_features
    
    def perform_segment_analysis(self) -> Dict[str, Any]:
        """
        Perform segment analysis comparing high vs low-selling wines.
        
        Returns:
        --------
        dict
            Segment analysis results
        """
        print("Performing high vs low-selling wines segment analysis...")
        
        target_data = self.df[self.target_column].dropna()
        
        # Define segments based on percentiles
        q33 = target_data.quantile(0.33)
        q67 = target_data.quantile(0.67)
        
        # Create segments
        self.df['sales_segment'] = 'Medium'
        self.df.loc[self.df[self.target_column] <= q33, 'sales_segment'] = 'Low'
        self.df.loc[self.df[self.target_column] >= q67, 'sales_segment'] = 'High'
        
        segments = {
            'Low': self.df[self.df['sales_segment'] == 'Low'],
            'Medium': self.df[self.df['sales_segment'] == 'Medium'], 
            'High': self.df[self.df['sales_segment'] == 'High']
        }
        
        segment_analysis = {
            'segment_definitions': {
                'Low': f'{self.target_column} ≤ {q33:.2f}',
                'Medium': f'{q33:.2f} < {self.target_column} < {q67:.2f}',
                'High': f'{self.target_column} ≥ {q67:.2f}'
            },
            'segment_sizes': {
                'Low': len(segments['Low']),
                'Medium': len(segments['Medium']),
                'High': len(segments['High'])
            },
            'segment_characteristics': {},
            'key_differentiators': [],
            'business_insights': []
        }
        
        # Analyze characteristics of each segment
        for segment_name, segment_df in segments.items():
            characteristics = {}
            
            for column in self.numerical_columns:
                if column in segment_df.columns and segment_df[column].notna().sum() > 0:
                    characteristics[column] = {
                        'mean': segment_df[column].mean(),
                        'median': segment_df[column].median(),
                        'std': segment_df[column].std(),
                        'count': segment_df[column].notna().sum()
                    }
            
            segment_analysis['segment_characteristics'][segment_name] = characteristics
        
        # Identify key differentiators between high and low segments
        differentiators = []
        
        if 'High' in segments and 'Low' in segments:
            high_segment = segments['High']
            low_segment = segments['Low']
            
            for column in self.numerical_columns:
                if (column in high_segment.columns and column in low_segment.columns and
                    high_segment[column].notna().sum() > 10 and low_segment[column].notna().sum() > 10):
                    
                    high_mean = high_segment[column].mean()
                    low_mean = low_segment[column].mean()
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((high_segment[column].var() * (len(high_segment) - 1)) + 
                                        (low_segment[column].var() * (len(low_segment) - 1))) / 
                                       (len(high_segment) + len(low_segment) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (high_mean - low_mean) / pooled_std
                        
                        # Statistical significance test
                        try:
                            t_stat, p_value = stats.ttest_ind(
                                high_segment[column].dropna(), 
                                low_segment[column].dropna()
                            )
                        except:
                            t_stat, p_value = np.nan, 1.0
                        
                        if abs(cohens_d) > 0.2 and p_value < 0.05:  # Meaningful and significant difference
                            differentiators.append({
                                'variable': column,
                                'high_segment_mean': high_mean,
                                'low_segment_mean': low_mean,
                                'difference': high_mean - low_mean,
                                'cohens_d': cohens_d,
                                'p_value': p_value,
                                'interpretation': self._interpret_segment_difference(column, cohens_d, high_mean, low_mean)
                            })
        
        # Sort by effect size
        differentiators.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
        segment_analysis['key_differentiators'] = differentiators
        
        # Generate business insights
        business_insights = self._generate_segment_insights(segment_analysis, differentiators)
        segment_analysis['business_insights'] = business_insights
        
        # Create segment analysis visualizations
        self._create_segment_plots(segments, differentiators)
        
        return segment_analysis
    
    def _interpret_segment_difference(self, variable: str, cohens_d: float, 
                                    high_mean: float, low_mean: float) -> str:
        """Interpret segment differences for business context."""
        
        direction = "higher" if cohens_d > 0 else "lower"
        magnitude = "substantially" if abs(cohens_d) > 0.8 else "moderately" if abs(cohens_d) > 0.5 else "slightly"
        
        variable_contexts = {
            'STARS': f"Top-selling wines have {magnitude} {direction} ratings ({high_mean:.2f} vs {low_mean:.2f})",
            'LabelAppeal': f"Top sellers have {magnitude} {direction} label appeal ({high_mean:.2f} vs {low_mean:.2f})",
            'Alcohol': f"Top sellers have {magnitude} {direction} alcohol content ({high_mean:.2f}% vs {low_mean:.2f}%)",
            'pH': f"Top sellers are {magnitude} {'less acidic' if cohens_d > 0 else 'more acidic'} ({high_mean:.2f} vs {low_mean:.2f} pH)",
            'ResidualSugar': f"Top sellers have {magnitude} {'more' if cohens_d > 0 else 'less'} residual sugar ({high_mean:.2f} vs {low_mean:.2f} g/L)"
        }
        
        return variable_contexts.get(variable, 
                                   f"Top sellers have {magnitude} {direction} {variable} ({high_mean:.2f} vs {low_mean:.2f})")
    
    def _generate_segment_insights(self, segment_analysis: Dict[str, Any], 
                                 differentiators: List[Dict[str, Any]]) -> List[str]:
        """Generate business insights from segment analysis."""
        
        insights = []
        
        # Segment size insights
        sizes = segment_analysis['segment_sizes']
        total_wines = sum(sizes.values())
        
        insights.append(f"Wine portfolio breakdown: {sizes['Low']} low sellers ({sizes['Low']/total_wines*100:.1f}%), "
                       f"{sizes['Medium']} medium sellers ({sizes['Medium']/total_wines*100:.1f}%), "
                       f"{sizes['High']} high sellers ({sizes['High']/total_wines*100:.1f}%)")
        
        # Top differentiators insights
        if differentiators:
            top_differentiator = differentiators[0]
            insights.append(f"Strongest differentiator: {top_differentiator['interpretation']}")
            
            # Quality vs chemistry insights
            quality_diffs = [d for d in differentiators if d['variable'] in self.quality_columns]
            chemistry_diffs = [d for d in differentiators if d['variable'] in self.chemical_columns]
            
            if quality_diffs and chemistry_diffs:
                insights.append("Both quality ratings and wine chemistry differentiate high vs low sellers")
            elif quality_diffs:
                insights.append("Quality ratings are the primary differentiators for wine sales success")
            elif chemistry_diffs:
                insights.append("Wine chemistry composition drives sales differences")
            
            # Actionable insights
            actionable_factors = [d for d in differentiators[:3] 
                                if d['variable'] in self.chemical_columns]
            
            if actionable_factors:
                insights.append("Actionable chemical composition factors for improving sales:")
                for factor in actionable_factors:
                    insights.append(f"  • {factor['interpretation']}")
        
        return insights
    
    def generate_business_insights(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive business insights from all analyses.
        
        Parameters:
        -----------
        all_results : dict
            Combined results from all analysis methods
            
        Returns:
        --------
        dict
            Comprehensive business insights and recommendations
        """
        print("Generating comprehensive business insights...")
        
        business_insights = {
            'executive_summary': [],
            'key_findings': [],
            'actionable_recommendations': [],
            'market_opportunities': [],
            'product_development_insights': [],
            'sales_strategy_recommendations': []
        }
        
        # Executive Summary
        distribution_stats = all_results.get('distribution_analysis', {})
        relationship_results = all_results.get('relationship_analysis', {})
        segment_results = all_results.get('segment_analysis', {})
        
        if distribution_stats:
            business_insights['executive_summary'].append(
                f"Wine sales range from {distribution_stats.get('min', 0)} to {distribution_stats.get('max', 0)} "
                f"with average sales of {distribution_stats.get('mean', 0):.1f}")
            
            business_insights['executive_summary'].append(
                f"Sales distribution is {distribution_stats.get('skewness_interpretation', 'unknown')}")
        
        if relationship_results and 'correlation_summary' in relationship_results:
            summary = relationship_results['correlation_summary']
            business_insights['executive_summary'].append(
                f"{summary.get('significant_relationships', 0)} variables significantly predict wine sales")
        
        # Key Findings
        if relationship_results and 'top_positive_predictors' in relationship_results:
            top_positive = relationship_results['top_positive_predictors'][:3]
            if top_positive:
                business_insights['key_findings'].append("Top sales drivers:")
                for predictor in top_positive:
                    business_insights['key_findings'].append(
                        f"  • {predictor['business_interpretation']}")
        
        if segment_results and 'key_differentiators' in segment_results:
            top_diffs = segment_results['key_differentiators'][:3]
            if top_diffs:
                business_insights['key_findings'].append("Key differences between high and low-selling wines:")
                for diff in top_diffs:
                    business_insights['key_findings'].append(f"  • {diff['interpretation']}")
        
        # Actionable Recommendations
        recommendations = []
        
        if relationship_results and 'top_positive_predictors' in relationship_results:
            # Focus on controllable factors
            controllable_predictors = [p for p in relationship_results['top_positive_predictors'] 
                                     if p['variable'] in self.chemical_columns]
            
            if controllable_predictors:
                recommendations.append("Optimize wine chemistry composition:")
                for predictor in controllable_predictors[:3]:
                    var = predictor['variable']
                    correlation = predictor['pearson_correlation']
                    if correlation > 0:
                        recommendations.append(f"  • Increase {var} to boost sales potential")
                    else:
                        recommendations.append(f"  • Reduce {var} to boost sales potential")
        
        if segment_results and 'business_insights' in segment_results:
            segment_insights = segment_results['business_insights']
            actionable_insights = [insight for insight in segment_insights if 'Actionable' in insight]
            recommendations.extend(actionable_insights)
        
        business_insights['actionable_recommendations'] = recommendations
        
        # Market Opportunities
        opportunities = []
        
        if distribution_stats:
            # Look for gaps in the market
            if distribution_stats.get('skewness', 0) > 0.5:
                opportunities.append("Market gap: Many wines underperform - opportunity to improve low sellers")
            elif distribution_stats.get('skewness', 0) < -0.5:
                opportunities.append("Market opportunity: Few wines in lower price/performance segments")
        
        # Zero-inflation insights
        zero_analysis = all_results.get('zero_inflation_analysis', {})
        if zero_analysis and zero_analysis.get('zero_proportion', 0) > 0.1:
            opportunities.append(f"Address non-selling wines: {zero_analysis['zero_proportion']*100:.1f}% of wines have zero sales")
        
        business_insights['market_opportunities'] = opportunities
        
        return business_insights
    
    def _create_distribution_plots(self, target_data: pd.Series, stats: Dict[str, Any]) -> None:
        """Create comprehensive distribution plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.target_column} Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Histogram with KDE
        axes[0, 0].hist(target_data, bins=30, alpha=0.7, density=True, edgecolor='black')
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
        axes[0, 0].axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.2f}")
        
        # Add KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(target_data.dropna())
            x_range = np.linspace(target_data.min(), target_data.max(), 100)
            axes[0, 0].plot(x_range, kde(x_range), color='blue', linewidth=2, label='KDE')
        except:
            pass
        
        axes[0, 0].set_xlabel('Wine Sales')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Sales Distribution with Central Tendencies')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Box plot with outliers
        box_plot = axes[0, 1].boxplot(target_data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        axes[0, 1].set_ylabel('Wine Sales')
        axes[0, 1].set_title('Sales Distribution Box Plot')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Value counts bar chart
        value_counts = stats['value_counts']
        axes[0, 2].bar(value_counts.index, value_counts.values, alpha=0.7)
        axes[0, 2].set_xlabel('Sales Value')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Sales Value Frequency')
        axes[0, 2].grid(alpha=0.3)
        
        # Plot 4: Cumulative distribution
        sorted_data = np.sort(target_data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 0].plot(sorted_data, cumulative_prob, linewidth=2)
        axes[1, 0].set_xlabel('Wine Sales')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Q-Q plot for normality check
        from scipy import stats as scipy_stats
        scipy_stats.probplot(target_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(alpha=0.3)
        
        # Plot 6: Statistics summary text
        axes[1, 2].axis('off')
        stats_text = f"""
        Distribution Statistics:
        
        Count: {stats['count']:,}
        Mean: {stats['mean']:.3f}
        Median: {stats['median']:.3f}
        Std Dev: {stats['std']:.3f}
        
        Range: {stats['min']:.1f} - {stats['max']:.1f}
        Q1: {stats['q1']:.3f}
        Q3: {stats['q3']:.3f}
        IQR: {stats['iqr']:.3f}
        
        Skewness: {stats['skewness']:.3f}
        Kurtosis: {stats['kurtosis']:.3f}
        CV: {stats['cv']:.1f}%
        
        {stats['skewness_interpretation']}
        """
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/target_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_zero_inflation_plot(self, zero_analysis: Dict[str, Any], target_data: pd.Series) -> None:
        """Create zero-inflation analysis plot."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Value distribution pie chart
        value_dist = zero_analysis['value_distribution']
        labels = ['Zero Sales', 'Low (1-2)', 'Moderate (3-5)', 'Good (6-8)', 'Excellent (>8)']
        sizes = [value_dist['zero_sales'], value_dist['low_sales_1_2'], 
                value_dist['moderate_sales_3_5'], value_dist['good_sales_6_8'], 
                value_dist['excellent_sales_above_8']]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Wine Sales Performance Distribution')
        
        # Plot 2: Histogram focusing on low values
        low_values = target_data[target_data <= 5]  # Focus on lower sales
        ax2.hist(low_values, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, 
                   label=f'Zero Sales: {zero_analysis["zero_count"]} wines')
        ax2.set_xlabel('Wine Sales (Low Range)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Zero-Inflation Analysis\n(Focus on Low Sales Range)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Zero-Inflation Level: {zero_analysis["zero_inflation_level"]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/zero_inflation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_relationship_plots(self, relationships: Dict[str, Any], target_data: pd.Series) -> None:
        """Create relationship analysis plots."""
        
        # Plot 1: Correlation heatmap focusing on TARGET
        all_relationships = relationships['all_relationships']
        
        if all_relationships:
            correlation_data = []
            variable_names = []
            
            for var, rel_data in all_relationships.items():
                correlation_data.append(rel_data['pearson_correlation'])
                variable_names.append(var)
            
            # Create horizontal bar plot of correlations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Sort by absolute correlation
            sorted_indices = sorted(range(len(correlation_data)), 
                                  key=lambda i: abs(correlation_data[i]), reverse=True)
            
            sorted_correlations = [correlation_data[i] for i in sorted_indices]
            sorted_names = [variable_names[i] for i in sorted_indices]
            
            colors = ['red' if corr < 0 else 'blue' for corr in sorted_correlations]
            
            bars = ax1.barh(range(len(sorted_correlations)), sorted_correlations, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(sorted_names)))
            ax1.set_yticklabels(sorted_names)
            ax1.set_xlabel('Pearson Correlation with Wine Sales')
            ax1.set_title('Variable Correlations with Wine Sales')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax1.grid(alpha=0.3)
            
            # Add significance indicators
            for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
                var_name = variable_names[idx]
                p_value = all_relationships[var_name]['pearson_p_value']
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = ''
                
                if significance:
                    x_pos = bar.get_width() + (0.02 if bar.get_width() > 0 else -0.02)
                    ax1.text(x_pos, bar.get_y() + bar.get_height()/2, significance,
                            ha='left' if bar.get_width() > 0 else 'right', va='center', fontweight='bold')
            
            # Plot 2: Top predictors scatter plots
            top_predictors = relationships['all_correlations_ranked'][:4]
            
            if len(top_predictors) >= 4:
                # Create 2x2 subplot for top 4 predictors
                fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                for i, predictor in enumerate(top_predictors):
                    var_name = predictor['variable']
                    correlation = predictor['pearson_correlation']
                    p_value = predictor['pearson_p_value']
                    
                    # Get clean paired data
                    paired_data = self.df[[self.target_column, var_name]].dropna()
                    
                    if len(paired_data) > 0:
                        x = paired_data[var_name]
                        y = paired_data[self.target_column]
                        
                        axes[i].scatter(x, y, alpha=0.6, s=30)
                        
                        # Add trend line
                        try:
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            axes[i].plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                        except:
                            pass
                        
                        axes[i].set_xlabel(var_name)
                        axes[i].set_ylabel('Wine Sales')
                        axes[i].set_title(f'{var_name}\nr = {correlation:.3f}, p = {p_value:.4f}')
                        axes[i].grid(alpha=0.3)
                
                plt.suptitle('Top 4 Predictive Features - Scatter Plots', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/top_predictors_scatter_plots.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save correlation bar plot
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_correlations)), sorted_correlations, color=colors, alpha=0.7)
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Pearson Correlation with Wine Sales')
            plt.title('Variable Correlations with Wine Sales\n*** p<0.001, ** p<0.01, * p<0.05')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(alpha=0.3)
            
            # Add significance indicators
            for i, idx in enumerate(sorted_indices):
                var_name = variable_names[idx]
                p_value = all_relationships[var_name]['pearson_p_value']
                correlation = sorted_correlations[i]
                
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = ''
                
                if significance:
                    x_pos = correlation + (0.02 if correlation > 0 else -0.02)
                    plt.text(x_pos, i, significance, ha='left' if correlation > 0 else 'right', 
                            va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_predictive_features_plot(self, predictive_analysis: Dict[str, Any]) -> None:
        """Create predictive features analysis plot."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Random Forest Feature Importance
        if 'random_forest' in predictive_analysis and 'error' not in predictive_analysis['random_forest']:
            rf_importance = predictive_analysis['random_forest']['feature_importance']
            top_rf = rf_importance.head(10)
            
            axes[0].barh(range(len(top_rf)), top_rf['importance'], alpha=0.7)
            axes[0].set_yticks(range(len(top_rf)))
            axes[0].set_yticklabels(top_rf['feature'])
            axes[0].set_xlabel('Feature Importance')
            axes[0].set_title('Random Forest\nFeature Importance')
            axes[0].grid(alpha=0.3)
        
        # Plot 2: Mutual Information Scores
        if 'mutual_information' in predictive_analysis and 'error' not in predictive_analysis['mutual_information']:
            mutual_info = predictive_analysis['mutual_information']['feature_scores']
            top_mutual = mutual_info.head(10)
            
            axes[1].barh(range(len(top_mutual)), top_mutual['mutual_info_score'], alpha=0.7, color='orange')
            axes[1].set_yticks(range(len(top_mutual)))
            axes[1].set_yticklabels(top_mutual['feature'])
            axes[1].set_xlabel('Mutual Information Score')
            axes[1].set_title('Mutual Information\nFeature Scores')
            axes[1].grid(alpha=0.3)
        
        # Plot 3: Consensus Ranking
        if 'consensus_ranking' in predictive_analysis:
            consensus = predictive_analysis['consensus_ranking']['ranking'][:10]
            features = [item[0] for item in consensus]
            scores = [item[1]['total_score'] for item in consensus]
            
            axes[2].barh(range(len(features)), scores, alpha=0.7, color='green')
            axes[2].set_yticks(range(len(features)))
            axes[2].set_yticklabels(features)
            axes[2].set_xlabel('Consensus Score')
            axes[2].set_title('Consensus Feature\nRanking')
            axes[2].grid(alpha=0.3)
        
        plt.suptitle('Predictive Features Analysis - Multiple Methods', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/predictive_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_segment_plots(self, segments: Dict[str, pd.DataFrame], 
                            differentiators: List[Dict[str, Any]]) -> None:
        """Create segment analysis plots."""
        
        # Plot 1: Segment characteristics comparison
        if len(differentiators) >= 6:
            top_differentiators = differentiators[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, diff in enumerate(top_differentiators):
                variable = diff['variable']
                
                # Create box plots for each segment
                segment_data = []
                segment_labels = []
                
                for segment_name, segment_df in segments.items():
                    if variable in segment_df.columns:
                        data = segment_df[variable].dropna()
                        if len(data) > 0:
                            segment_data.append(data)
                            segment_labels.append(f"{segment_name}\n(n={len(data)})")
                
                if segment_data:
                    box_plot = axes[i].boxplot(segment_data, labels=segment_labels, patch_artist=True)
                    
                    # Color boxes
                    colors = ['red', 'yellow', 'green']
                    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    axes[i].set_title(f'{variable}\n(Effect size: {diff["cohens_d"]:.3f})')
                    axes[i].set_ylabel(variable)
                    axes[i].grid(alpha=0.3)
                    
                    # Add statistical significance indicator
                    if diff['p_value'] < 0.001:
                        axes[i].text(0.5, 0.95, '***', transform=axes[i].transAxes, 
                                   ha='center', va='top', fontweight='bold', fontsize=14)
                    elif diff['p_value'] < 0.01:
                        axes[i].text(0.5, 0.95, '**', transform=axes[i].transAxes, 
                                   ha='center', va='top', fontweight='bold', fontsize=14)
                    elif diff['p_value'] < 0.05:
                        axes[i].text(0.5, 0.95, '*', transform=axes[i].transAxes, 
                                   ha='center', va='top', fontweight='bold', fontsize=14)
            
            plt.suptitle('Key Differentiators: High vs Medium vs Low-Selling Wines\n*** p<0.001, ** p<0.01, * p<0.05', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/segment_analysis_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Effect size summary
        if differentiators:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            variables = [diff['variable'] for diff in differentiators[:10]]
            effect_sizes = [abs(diff['cohens_d']) for diff in differentiators[:10]]
            colors = ['green' if diff['cohens_d'] > 0 else 'red' for diff in differentiators[:10]]
            
            bars = ax.barh(range(len(variables)), effect_sizes, color=colors, alpha=0.7)
            ax.set_yticks(range(len(variables)))
            ax.set_yticklabels(variables)
            ax.set_xlabel('Effect Size (|Cohen\'s d|)')
            ax.set_title('Segment Differentiators - Effect Sizes\nGreen: Higher in top sellers, Red: Lower in top sellers')
            ax.grid(alpha=0.3)
            
            # Add effect size interpretation lines
            ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.7, label='Small effect')
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
            ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.7, label='Large effect')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/effect_sizes_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

def comprehensive_target_analysis(df: pd.DataFrame, target_column: str = 'TARGET',
                                 output_dir: str = 'target_analysis_plots') -> Dict[str, Any]:
    """
    Perform comprehensive TARGET variable analysis for wine sales insights.
    
    This function provides specialized analysis of the wine sales (TARGET) variable
    including distribution analysis, zero-inflation assessment, relationship exploration,
    predictive feature identification, and business-focused segment analysis.
    
    Analysis Components:
    -------------------
    1. Distribution Analysis: Comprehensive statistical characterization
    2. Zero-Inflation Analysis: Assessment of non-selling wines  
    3. Relationship Analysis: Correlations with all chemical/quality variables
    4. Predictive Features: Multi-method identification of sales drivers
    5. Segment Analysis: High vs low-selling wine characteristics
    6. Business Insights: Actionable recommendations for wine sales
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wine dataset with TARGET variable and chemical composition data
    target_column : str, default 'TARGET'
        Name of the wine sales target variable
    output_dir : str, default 'target_analysis_plots'
        Directory to save business visualization plots
        
    Returns:
    --------
    dict
        Comprehensive TARGET analysis results including:
        - Distribution statistics and business interpretations
        - Zero-inflation assessment with market implications
        - Relationship analysis with chemical composition variables
        - Multi-method predictive feature identification
        - High vs low-selling wine segment comparison
        - Executive business insights and actionable recommendations
        
    Generated Visualizations:
    -------------------------
    - target_distribution_analysis.png: 6-panel distribution characterization
    - zero_inflation_analysis.png: Non-selling wines assessment
    - correlation_analysis.png: Sales correlations with all variables
    - top_predictors_scatter_plots.png: Key relationship visualizations
    - predictive_features_analysis.png: Multi-method feature importance
    - segment_analysis_comparison.png: High vs low seller characteristics
    - effect_sizes_analysis.png: Statistical significance of differences
    
    Business Focus:
    ---------------
    - Identifies chemical properties that drive wine sales success
    - Provides actionable insights for product development
    - Highlights market opportunities and gaps
    - Delivers executive-level business recommendations
    """
    
    print("="*70)
    print("COMPREHENSIVE TARGET VARIABLE ANALYSIS")
    print("Wine Sales Business Intelligence System")
    print("="*70)
    
    # Initialize analyzer
    analyzer = TargetAnalyzer(df, target_column, output_dir)
    
    # Comprehensive analysis pipeline
    print(f"\n🎯 ANALYZING {target_column} VARIABLE")
    print("-" * 50)
    
    # Step 1: Distribution Analysis
    distribution_analysis = analyzer.analyze_target_distribution()
    
    # Step 2: Zero-Inflation Analysis  
    zero_inflation_analysis = analyzer.analyze_zero_inflation()
    
    # Step 3: Relationship Analysis
    relationship_analysis = analyzer.analyze_target_relationships()
    
    # Step 4: Predictive Features
    predictive_analysis = analyzer.identify_predictive_features()
    
    # Step 5: Segment Analysis
    segment_analysis = analyzer.perform_segment_analysis()
    
    # Step 6: Business Insights
    all_results = {
        'distribution_analysis': distribution_analysis,
        'zero_inflation_analysis': zero_inflation_analysis,
        'relationship_analysis': relationship_analysis,
        'predictive_analysis': predictive_analysis,
        'segment_analysis': segment_analysis
    }
    
    business_insights = analyzer.generate_business_insights(all_results)
    
    # Compile final results
    final_results = {
        **all_results,
        'business_insights': business_insights,
        'dataset_summary': {
            'total_wines': len(df),
            'target_variable': target_column,
            'chemical_variables': len(analyzer.chemical_columns),
            'quality_variables': len(analyzer.quality_columns),
            'analysis_completeness': 'Full analysis completed successfully'
        }
    }
    
    # Print executive summary
    print(f"\n📊 EXECUTIVE SUMMARY:")
    print(f"   • Dataset: {len(df):,} wines analyzed")
    print(f"   • Sales range: {distribution_analysis.get('min', 0):.0f} - {distribution_analysis.get('max', 0):.0f}")
    print(f"   • Average sales: {distribution_analysis.get('mean', 0):.1f}")
    print(f"   • Zero-sales wines: {zero_inflation_analysis.get('zero_count', 0):,} ({zero_inflation_analysis.get('zero_proportion', 0)*100:.1f}%)")
    
    if relationship_analysis and 'correlation_summary' in relationship_analysis:
        summary = relationship_analysis['correlation_summary']
        print(f"   • Significant predictors: {summary.get('significant_relationships', 0)} variables")
    
    print(f"\n🎯 TOP SALES DRIVERS:")
    if relationship_analysis and 'top_positive_predictors' in relationship_analysis:
        top_predictors = relationship_analysis['top_positive_predictors'][:3]
        for i, predictor in enumerate(top_predictors, 1):
            print(f"   {i}. {predictor['variable']}: r = {predictor['pearson_correlation']:.3f}")
    
    print(f"\n🔍 SEGMENT INSIGHTS:")
    if segment_analysis and 'key_differentiators' in segment_analysis:
        top_diffs = segment_analysis['key_differentiators'][:3]
        for i, diff in enumerate(top_diffs, 1):
            direction = "↑" if diff['cohens_d'] > 0 else "↓"
            print(f"   {i}. {diff['variable']}: {direction} High sellers have {abs(diff['cohens_d']):.2f} effect size")
    
    print(f"\n💡 KEY BUSINESS RECOMMENDATIONS:")
    if business_insights and 'actionable_recommendations' in business_insights:
        recommendations = business_insights['actionable_recommendations'][:5]
        for i, rec in enumerate(recommendations, 1):
            if not rec.startswith('  •'):  # Skip sub-bullets
                print(f"   {i}. {rec}")
    
    print(f"\n📁 Business visualizations saved to: {output_dir}/")
    print(f"   • 7 executive-level analysis charts generated")
    
    print("\n" + "="*70)
    print("TARGET ANALYSIS COMPLETE - BUSINESS INSIGHTS READY")
    print("="*70)
    
    return final_results

# Example usage
if __name__ == "__main__":
    # Load wine dataset for testing
    print("Loading wine dataset...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Perform comprehensive TARGET analysis
    analysis_results = comprehensive_target_analysis(df, target_column='TARGET')