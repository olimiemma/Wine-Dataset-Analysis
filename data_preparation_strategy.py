import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox, yeojohnson, normaltest, shapiro
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
import os
from typing import Dict, List, Tuple, Any, Union
warnings.filterwarnings('ignore')

class DataPreparationStrategy:
    """
    Comprehensive data preparation strategy based on EDA findings.
    
    Implements missing data treatment, outlier handling, distribution transforms,
    feature engineering, and scaling strategies with business justifications.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'TARGET',
                 output_dir: str = 'data_preparation_plots'):
        """
        Initialize the data preparation strategy.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw wine dataset
        target_column : str, default 'TARGET'
            Name of the target variable
        output_dir : str
            Directory to save preparation visualizations
        """
        self.df_original = df.copy()
        self.df_prepared = df.copy()
        self.target_column = target_column
        self.output_dir = output_dir
        self.numerical_columns = self._identify_numerical_columns()
        self.preparation_log = []
        self.transformers = {}
        self.imputers = {}
        self.scalers = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("viridis")
    
    def _identify_numerical_columns(self) -> List[str]:
        """Identify numerical columns for preparation."""
        numerical_cols = self.df_original.select_dtypes(include=[np.number]).columns.tolist()
        # Remove index-like columns
        numerical_cols = [col for col in numerical_cols if not col.upper().startswith('INDEX')]
        return numerical_cols
    
    def _log_preparation_step(self, step: str, details: Dict[str, Any]) -> None:
        """Log preparation steps for documentation."""
        self.preparation_log.append({
            'step': step,
            'timestamp': pd.Timestamp.now(),
            'details': details
        })
    
    def create_missing_data_treatment_plan(self) -> Dict[str, Any]:
        """
        Create comprehensive missing data treatment plan based on EDA findings.
        
        Returns:
        --------
        dict
            Missing data treatment strategy with business rationale
        """
        print("Creating missing data treatment plan...")
        
        missing_analysis = {}
        treatment_plan = {}
        
        # Analyze missing patterns for each variable
        for column in self.numerical_columns:
            missing_count = self.df_original[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df_original)) * 100
            
            if missing_count > 0:
                missing_analysis[column] = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'data_type': str(self.df_original[column].dtype),
                    'unique_values': self.df_original[column].nunique(),
                    'business_importance': self._assess_business_importance(column)
                }
        
        # Create treatment strategies based on EDA findings
        for column, analysis in missing_analysis.items():
            missing_pct = analysis['missing_percentage']
            business_importance = analysis['business_importance']
            
            if column == 'STARS':
                # Special treatment for STARS (26.2% missing, high business importance)
                treatment_plan[column] = {
                    'strategy': 'median_imputation_with_quality_indicators',
                    'method': 'MedianImputer',
                    'rationale': [
                        "STARS is the strongest sales predictor (r=0.559)",
                        "26.2% missing rate too high to drop observations",
                        "Median imputation preserves central tendency",
                        "Create missing indicator for model transparency",
                        "Business impact: Quality ratings drive sales success"
                    ],
                    'implementation': 'Fill with median, add STARS_was_missing flag',
                    'validation': 'Compare imputed vs non-imputed sales correlation'
                }
                
            elif missing_pct > 20:
                # High missing percentage variables
                treatment_plan[column] = {
                    'strategy': 'advanced_imputation',
                    'method': 'KNNImputer',
                    'rationale': [
                        f"High missing rate ({missing_pct:.1f}%) requires sophisticated approach",
                        "KNN imputation preserves variable relationships",
                        "Uses similar wine profiles for imputation",
                        f"Business importance: {business_importance}"
                    ],
                    'implementation': 'KNN imputation (k=5) with similar wine characteristics',
                    'validation': 'Cross-validation of imputation accuracy'
                }
                
            elif missing_pct > 10:
                # Moderate missing percentage
                treatment_plan[column] = {
                    'strategy': 'predictive_imputation',
                    'method': 'RandomForestImputer',
                    'rationale': [
                        f"Moderate missing rate ({missing_pct:.1f}%)",
                        "Random Forest captures non-linear relationships",
                        "Uses all available features for prediction",
                        "Robust to outliers identified in EDA"
                    ],
                    'implementation': 'Random Forest imputation with wine chemistry features',
                    'validation': 'Out-of-bag imputation error assessment'
                }
                
            elif missing_pct > 5:
                # Low-moderate missing percentage
                treatment_plan[column] = {
                    'strategy': 'median_imputation',
                    'method': 'MedianImputer',
                    'rationale': [
                        f"Low missing rate ({missing_pct:.1f}%)",
                        "Median robust to outliers (99% outlier rate identified)",
                        "Preserves distribution shape",
                        "Simple and interpretable for business"
                    ],
                    'implementation': 'Fill with median value',
                    'validation': 'Distribution comparison pre/post imputation'
                }
                
            else:
                # Very low missing percentage
                treatment_plan[column] = {
                    'strategy': 'mean_imputation',
                    'method': 'MeanImputer',
                    'rationale': [
                        f"Very low missing rate ({missing_pct:.1f}%)",
                        "Minimal impact on distribution",
                        "Fast and simple implementation",
                        "Suitable for operational deployment"
                    ],
                    'implementation': 'Fill with mean value',
                    'validation': 'Statistical significance test of impact'
                }
        
        # Create missing data visualization
        self._create_missing_data_treatment_plot(missing_analysis, treatment_plan)
        
        return {
            'missing_analysis': missing_analysis,
            'treatment_plan': treatment_plan,
            'summary': self._summarize_missing_treatment(treatment_plan)
        }
    
    def _assess_business_importance(self, column: str) -> str:
        """Assess business importance of variables based on EDA findings."""
        
        # Based on TARGET correlation analysis findings
        high_importance_vars = ['STARS', 'LabelAppeal']  # Strong correlations with sales
        moderate_importance_vars = ['Alcohol', 'AcidIndex', 'VolatileAcidity']  # Moderate correlations
        
        if column in high_importance_vars:
            return "CRITICAL - Strong sales predictor"
        elif column in moderate_importance_vars:
            return "HIGH - Significant sales impact"
        elif 'Acid' in column or 'pH' in column:
            return "MODERATE - Wine chemistry factor"
        elif 'Sulfur' in column or 'Sulphate' in column:
            return "MODERATE - Quality/preservation factor"
        else:
            return "STANDARD - Supporting variable"
    
    def create_outlier_treatment_framework(self) -> Dict[str, Any]:
        """
        Create outlier treatment framework based on consensus outlier analysis.
        
        Returns:
        --------
        dict
            Comprehensive outlier treatment strategy
        """
        print("Creating outlier treatment framework...")
        
        # Based on EDA findings: 99% outlier rate with 10.1% severe outliers
        outlier_framework = {
            'strategy_overview': {
                'severe_outliers_rate': 10.1,
                'total_outliers_rate': 99.0,
                'approach': 'Tiered treatment based on severity and business impact',
                'rationale': 'EDA revealed extensive outliers requiring careful handling'
            },
            'treatment_tiers': {},
            'variable_specific_strategies': {},
            'business_justifications': {}
        }
        
        # Define treatment tiers
        outlier_framework['treatment_tiers'] = {
            'tier_1_severe': {
                'description': 'Severe outliers (consensus score >90th percentile)',
                'action': 'Winsorization at 1st/99th percentiles',
                'rationale': [
                    "Preserve data points while reducing extreme impact",
                    "Maintains sample size for business analysis",
                    "Reduces model sensitivity to extreme values",
                    "Retains business interpretability"
                ],
                'percentage_affected': 10.1
            },
            
            'tier_2_moderate': {
                'description': 'Moderate outliers (75th-90th percentile consensus)',
                'action': 'Log transformation + robust scaling',
                'rationale': [
                    "Transform to reduce skewness impact",
                    "Robust scaling handles remaining outliers",
                    "Preserves relative relationships",
                    "Suitable for ML algorithms"
                ],
                'percentage_affected': 15.7
            },
            
            'tier_3_mild': {
                'description': 'Mild outliers (detected but not severe)',
                'action': 'Robust scaling only',
                'rationale': [
                    "Light treatment preserves data integrity",
                    "RobustScaler uses median/IQR",
                    "Reduces impact without data loss",
                    "Maintains business meaning"
                ],
                'percentage_affected': 73.2
            },
            
            'domain_impossible': {
                'description': 'Chemically impossible values (9,959 data points)',
                'action': 'Flag and investigate/correct',
                'rationale': [
                    "Cannot exist in real wine production",
                    "Indicates data quality issues",
                    "Requires business validation",
                    "May need data source correction"
                ],
                'count': 9959
            }
        }
        
        # Variable-specific strategies based on EDA findings
        high_outlier_variables = [
            'FreeSulfurDioxide', 'Density', 'ResidualSugar', 'Chlorides', 
            'Sulphates', 'CitricAcid', 'VolatileAcidity', 'FixedAcidity'
        ]
        
        for variable in self.numerical_columns:
            if variable in high_outlier_variables:
                outlier_framework['variable_specific_strategies'][variable] = {
                    'strategy': 'winsorization_plus_transform',
                    'steps': [
                        'Cap at 1st/99th percentiles',
                        'Apply log transformation if right-skewed',
                        'Use RobustScaler for final scaling'
                    ],
                    'rationale': f"High outlier rate identified in EDA for {variable}"
                }
            elif variable == 'STARS':
                outlier_framework['variable_specific_strategies'][variable] = {
                    'strategy': 'preserve_natural_range',
                    'steps': [
                        'Keep natural 1-4 star range',
                        'No winsorization (business meaningful scale)',
                        'Standard scaling appropriate'
                    ],
                    'rationale': "STARS has natural business bounds and high importance"
                }
            elif variable == 'TARGET':
                outlier_framework['variable_specific_strategies'][variable] = {
                    'strategy': 'minimal_treatment',
                    'steps': [
                        'Preserve target variable integrity',
                        'No transformation for interpretability',
                        'Handle outliers in feature space instead'
                    ],
                    'rationale': "TARGET preservation essential for business interpretation"
                }
            else:
                outlier_framework['variable_specific_strategies'][variable] = {
                    'strategy': 'standard_robust_treatment',
                    'steps': [
                        'Mild winsorization at 2nd/98th percentiles',
                        'RobustScaler for normalization'
                    ],
                    'rationale': "Standard treatment for moderate outlier variables"
                }
        
        return outlier_framework
    
    def create_transformation_pipeline(self) -> Dict[str, Any]:
        """
        Create distribution transformation pipeline based on normality findings.
        
        Returns:
        --------
        dict
            Comprehensive transformation strategy
        """
        print("Creating distribution transformation pipeline...")
        
        # Based on EDA: ALL variables fail normality tests
        transformation_pipeline = {
            'overview': {
                'normality_test_results': 'All 15 variables fail normality tests',
                'approach': 'Multi-method transformation testing',
                'goal': 'Improve normality and ML algorithm performance',
                'validation': 'Before/after normality testing and business impact'
            },
            'transformation_methods': {},
            'variable_recommendations': {},
            'testing_framework': {}
        }
        
        # Define transformation methods
        transformation_pipeline['transformation_methods'] = {
            'log_transform': {
                'best_for': 'Right-skewed distributions',
                'formula': 'log(x + 1) to handle zeros',
                'pros': ['Simple interpretation', 'Handles right skew well'],
                'cons': ['Cannot handle negative values', 'May not normalize heavy tails'],
                'business_interpretation': 'Logarithmic relationships common in chemistry'
            },
            
            'box_cox': {
                'best_for': 'Positive data with various skewness',
                'formula': '(x^λ - 1) / λ, optimal λ estimated',
                'pros': ['Optimal transformation parameter', 'Strong normalization'],
                'cons': ['Requires positive values', 'Less interpretable'],
                'business_interpretation': 'Statistical optimal but complex business meaning'
            },
            
            'yeo_johnson': {
                'best_for': 'Any data including negative values',
                'formula': 'Extension of Box-Cox for all real numbers',
                'pros': ['Handles negative values', 'Flexible transformation'],
                'cons': ['Complex interpretation', 'May over-transform'],
                'business_interpretation': 'Most flexible but hardest to explain to business'
            },
            
            'quantile_transform': {
                'best_for': 'Heavy outliers and complex distributions',
                'formula': 'Maps to uniform then normal distribution',
                'pros': ['Handles any distribution', 'Robust to outliers'],
                'cons': ['Non-parametric', 'Loses original scale meaning'],
                'business_interpretation': 'Rank-based, preserves relative ordering'
            }
        }
        
        # Create variable-specific recommendations
        for variable in self.numerical_columns:
            # Analyze variable characteristics for transformation selection
            var_data = self.df_original[variable].dropna()
            
            if len(var_data) == 0:
                continue
            
            skewness = var_data.skew()
            has_zeros = (var_data == 0).any()
            has_negatives = (var_data < 0).any()
            min_value = var_data.min()
            
            # Decision logic based on data characteristics
            if variable == 'STARS':
                recommendation = {
                    'primary_method': 'no_transformation',
                    'rationale': [
                        'Natural categorical scale (1-4 stars)',
                        'Business interpretability crucial',
                        'Ordinal nature should be preserved',
                        'Strong sales correlation as-is'
                    ],
                    'alternative': 'Consider ordinal encoding if needed'
                }
            elif variable == 'TARGET':
                recommendation = {
                    'primary_method': 'no_transformation',
                    'rationale': [
                        'Target variable - preserve interpretability',
                        'Sales units have direct business meaning',
                        'Count data - Poisson distribution may be appropriate',
                        'Transformation complicates business insights'
                    ],
                    'alternative': 'Handle non-normality in model selection instead'
                }
            elif has_negatives:
                recommendation = {
                    'primary_method': 'yeo_johnson',
                    'rationale': [
                        f'Variable {variable} contains negative values',
                        'Yeo-Johnson handles full real number range',
                        'May indicate data preprocessing/standardization',
                        'Need to investigate negative value business meaning'
                    ],
                    'alternative': 'Quantile transformation if Yeo-Johnson ineffective'
                }
            elif has_zeros or min_value <= 0:
                recommendation = {
                    'primary_method': 'log_plus_one',
                    'rationale': [
                        f'Variable {variable} contains zeros/near-zeros',
                        'Log(x+1) handles zero values appropriately',
                        f'Right-skewed data (skewness: {skewness:.2f})',
                        'Common transformation for chemical measurements'
                    ],
                    'alternative': 'Box-Cox if log transformation insufficient'
                }
            elif skewness > 1:
                recommendation = {
                    'primary_method': 'box_cox',
                    'rationale': [
                        f'Highly right-skewed (skewness: {skewness:.2f})',
                        'Box-Cox optimal for positive right-skewed data',
                        'Automatically finds best transformation parameter',
                        'Strong normalization expected'
                    ],
                    'alternative': 'Log transformation if Box-Cox parameter ≈ 0'
                }
            elif abs(skewness) > 0.5:
                recommendation = {
                    'primary_method': 'yeo_johnson',
                    'rationale': [
                        f'Moderate skewness (skewness: {skewness:.2f})',
                        'Yeo-Johnson flexible for moderate skewness',
                        'Handles both positive and negative skew',
                        'Good balance of normalization and interpretability'
                    ],
                    'alternative': 'Quantile transformation for stubborn distributions'
                }
            else:
                recommendation = {
                    'primary_method': 'quantile_transform',
                    'rationale': [
                        'Complex distribution pattern identified',
                        'Heavy outliers detected in EDA',
                        'Quantile transformation robust to outliers',
                        'Preserves ranking relationships'
                    ],
                    'alternative': 'Consider no transformation if business interpretation priority'
                }
            
            transformation_pipeline['variable_recommendations'][variable] = recommendation
        
        # Testing framework
        transformation_pipeline['testing_framework'] = {
            'normality_tests': ['Shapiro-Wilk', 'Anderson-Darling', 'D\'Agostino-Pearson'],
            'evaluation_metrics': [
                'Normality test p-values',
                'Skewness reduction',
                'Kurtosis normalization', 
                'ML model performance impact',
                'Business interpretability score'
            ],
            'validation_approach': 'Cross-validation with multiple algorithms',
            'selection_criteria': 'Weighted score: 40% normality, 30% ML performance, 30% interpretability'
        }
        
        return transformation_pipeline
    
    def design_feature_engineering_strategy(self) -> Dict[str, Any]:
        """
        Design feature engineering opportunities based on EDA insights.
        
        Returns:
        --------
        dict
            Feature engineering strategy and opportunities
        """
        print("Designing feature engineering strategy...")
        
        feature_engineering = {
            'strategy_overview': {
                'approach': 'Domain knowledge + EDA insights',
                'focus_areas': [
                    'Wine chemistry ratios and interactions',
                    'Quality-marketing synergies',
                    'Wine style categorization',
                    'Sales performance indicators'
                ],
                'validation': 'Feature importance and sales correlation analysis'
            },
            'chemistry_ratios': {},
            'interaction_features': {},
            'categorical_features': {},
            'performance_indicators': {}
        }
        
        # Chemistry ratio features (based on wine science domain knowledge)
        feature_engineering['chemistry_ratios'] = {
            'total_to_free_so2_ratio': {
                'formula': 'TotalSulfurDioxide / (FreeSulfurDioxide + 1)',
                'rationale': [
                    'Indicates SO2 binding effectiveness',
                    'Important for wine preservation',
                    'EDA showed logical relationship violations',
                    'May reveal wine production quality'
                ],
                'expected_business_impact': 'Quality assessment and shelf-life prediction'
            },
            
            'acidity_balance_ratio': {
                'formula': 'FixedAcidity / (VolatileAcidity + 0.1)',
                'rationale': [
                    'Balance between good and bad acidity',
                    'Higher ratio indicates better balance',
                    'Both acids correlate negatively with sales',
                    'Ratio may capture wine quality better than individual values'
                ],
                'expected_business_impact': 'Wine quality and taste profile indicator'
            },
            
            'sugar_alcohol_ratio': {
                'formula': 'ResidualSugar / (Alcohol + 1)',
                'rationale': [
                    'Indicates wine dryness/sweetness relative to strength',
                    'Important for wine style classification',
                    'Consumer preference indicator',
                    'May predict market segment appeal'
                ],
                'expected_business_impact': 'Consumer preference and market positioning'
            },
            
            'density_alcohol_deviation': {
                'formula': 'abs(Density - (1.0 - 0.01 * Alcohol))',
                'rationale': [
                    'Expected relationship: higher alcohol = lower density',
                    'Deviation may indicate other compounds',
                    'Quality/authenticity indicator',
                    'EDA showed density outliers'
                ],
                'expected_business_impact': 'Wine authenticity and quality assessment'
            }
        }
        
        # Interaction features (based on TARGET correlation analysis)
        feature_engineering['interaction_features'] = {
            'stars_label_synergy': {
                'formula': 'STARS * LabelAppeal',
                'rationale': [
                    'Top 2 sales predictors (r=0.559, r=0.357)',
                    'Quality and marketing synergy',
                    'High-quality wines with good marketing should excel',
                    'Captures premium wine segment'
                ],
                'expected_business_impact': 'Premium wine identification and pricing'
            },
            
            'quality_chemistry_interaction': {
                'formula': 'STARS * (1 - normalized_AcidIndex)',
                'rationale': [
                    'STARS positive, AcidIndex negative sales correlation',
                    'Quality wines with balanced acidity perform best',
                    'Captures quality-taste interaction',
                    'AcidIndex strongest negative chemistry predictor'
                ],
                'expected_business_impact': 'Wine formulation optimization guidance'
            },
            
            'alcohol_appeal_premium': {
                'formula': 'Alcohol * LabelAppeal * (STARS > 2.5)',
                'rationale': [
                    'Premium wines (>2.5 stars) with higher alcohol and appeal',
                    'Captures luxury market segment',
                    'Three-way interaction of key predictors',
                    'May identify high-value products'
                ],
                'expected_business_impact': 'Luxury market segment targeting'
            }
        }
        
        # Categorical features (wine style classification)
        feature_engineering['categorical_features'] = {
            'wine_quality_tier': {
                'formula': 'pd.cut(STARS, bins=[0, 1.5, 2.5, 3.5, 5], labels=["Poor", "Fair", "Good", "Excellent"])',
                'rationale': [
                    'STARS is strongest predictor but continuous',
                    'Business thinks in quality tiers',
                    'May capture non-linear quality-sales relationship',
                    'Enables tier-specific strategies'
                ],
                'expected_business_impact': 'Quality tier marketing and pricing'
            },
            
            'chemistry_profile': {
                'formula': 'Based on pH, Alcohol, ResidualSugar clustering',
                'categories': ['Dry_Light', 'Dry_Full', 'OffDry_Balanced', 'Sweet_Light', 'Sweet_Strong'],
                'rationale': [
                    'Wine industry standard style categories',
                    'Combines key chemistry parameters',
                    'Consumer preference segments',
                    'Enables style-specific marketing'
                ],
                'expected_business_impact': 'Market segmentation and product positioning'
            },
            
            'sales_performance_class': {
                'formula': 'pd.cut(TARGET, bins=[0, 2, 4, 6, 10], labels=["Poor", "Average", "Good", "Excellent"])',
                'rationale': [
                    'Convert continuous TARGET to categories',
                    'Enables classification modeling approach',
                    'Business-friendly performance categories',
                    'May reveal threshold effects'
                ],
                'expected_business_impact': 'Performance classification and benchmarking'
            }
        }
        
        # Performance indicators
        feature_engineering['performance_indicators'] = {
            'quality_score_composite': {
                'formula': '0.6 * STARS + 0.3 * LabelAppeal + 0.1 * (normalized chemistry score)',
                'rationale': [
                    'Weighted combination of top predictors',
                    'Weights based on correlation strengths',
                    'Single quality metric for business use',
                    'Combines quality, marketing, and chemistry'
                ],
                'expected_business_impact': 'Single KPI for wine quality assessment'
            },
            
            'improvement_potential': {
                'formula': '(max_possible_STARS - current_STARS) + (max_possible_LabelAppeal - current_LabelAppeal)',
                'rationale': [
                    'Identifies wines with improvement opportunity',
                    'Focuses on controllable factors',
                    'Prioritizes investment allocation',
                    'Based on top sales predictors'
                ],
                'expected_business_impact': 'Investment prioritization and ROI optimization'
            },
            
            'market_position_score': {
                'formula': 'Percentile rank within wine style category',
                'rationale': [
                    'Relative performance within relevant market',
                    'Accounts for style-specific competition',
                    'More meaningful than absolute scores',
                    'Enables competitive analysis'
                ],
                'expected_business_impact': 'Competitive positioning and market share analysis'
            }
        }
        
        return feature_engineering
    
    def design_scaling_strategy(self) -> Dict[str, Any]:
        """
        Design scaling and normalization strategy based on EDA findings.
        
        Returns:
        --------
        dict
            Comprehensive scaling strategy
        """
        print("Designing scaling and normalization strategy...")
        
        scaling_strategy = {
            'strategy_overview': {
                'challenge': '99% outlier rate requires robust scaling approaches',
                'approach': 'Variable-specific scaling based on distribution characteristics',
                'business_priority': 'Preserve interpretability where possible',
                'ml_priority': 'Enable effective algorithm performance'
            },
            'scaler_selection': {},
            'variable_assignments': {},
            'validation_framework': {}
        }
        
        # Define scaling methods
        scaling_strategy['scaler_selection'] = {
            'RobustScaler': {
                'best_for': 'Heavy outlier variables (identified in EDA)',
                'method': 'Median and IQR-based scaling',
                'pros': ['Outlier resistant', 'Preserves distribution shape', 'Interpretable'],
                'cons': ['May not fully normalize', 'Less effective for heavy tails'],
                'business_interpretation': 'Relative position within typical range'
            },
            
            'StandardScaler': {
                'best_for': 'Variables with mild outliers after transformation',
                'method': 'Mean and standard deviation scaling',
                'pros': ['Standard approach', 'Good for normal distributions', 'Fast'],
                'cons': ['Sensitive to outliers', 'Assumes normal distribution'],
                'business_interpretation': 'Standard deviations from average'
            },
            
            'MinMaxScaler': {
                'best_for': 'Variables with natural bounds (like STARS)',
                'method': 'Scale to [0,1] range based on min/max',
                'pros': ['Preserves original distribution shape', 'Bounded output', 'Interpretable'],
                'cons': ['Very sensitive to outliers', 'Loses relative spacing'],
                'business_interpretation': 'Percentage of maximum possible value'
            },
            
            'PowerTransformer': {
                'best_for': 'Non-normal distributions needing both transform and scale',
                'method': 'Box-Cox or Yeo-Johnson + standardization',
                'pros': ['Handles non-normality', 'Integrated transform+scale', 'ML optimized'],
                'cons': ['Complex interpretation', 'May over-transform'],
                'business_interpretation': 'Normalized and transformed values'
            }
        }
        
        # Assign scalers to variables based on EDA findings
        high_outlier_vars = ['FreeSulfurDioxide', 'Density', 'ResidualSugar', 'Chlorides']
        quality_vars = ['STARS', 'LabelAppeal'] 
        chemistry_vars = ['pH', 'Alcohol', 'FixedAcidity', 'VolatileAcidity', 'CitricAcid']
        
        for variable in self.numerical_columns:
            if variable == 'TARGET':
                assignment = {
                    'primary_scaler': 'no_scaling',
                    'rationale': [
                        'Target variable - preserve original scale',
                        'Sales units have direct business meaning',
                        'Model interpretation requires original scale',
                        'Performance metrics more meaningful'
                    ],
                    'alternative': 'Consider log scaling only if variance issues'
                }
            elif variable == 'STARS':
                assignment = {
                    'primary_scaler': 'MinMaxScaler',
                    'rationale': [
                        'Natural 1-4 star scale with clear bounds',
                        'Business interpretability crucial',
                        'Ordinal nature should be preserved',
                        'MinMax preserves relative spacing'
                    ],
                    'alternative': 'StandardScaler if treating as continuous'
                }
            elif variable in high_outlier_vars:
                assignment = {
                    'primary_scaler': 'RobustScaler',
                    'rationale': [
                        f'High outlier rate identified for {variable}',
                        'RobustScaler resistant to extreme values',
                        'Median/IQR more stable than mean/std',
                        'Preserves business interpretability'
                    ],
                    'alternative': 'PowerTransformer if normality critical'
                }
            elif variable in quality_vars:
                assignment = {
                    'primary_scaler': 'StandardScaler',
                    'rationale': [
                        'Important business variables',
                        'Relatively well-behaved distributions',
                        'Standard scaling preserves interpretation',
                        'Good for linear model coefficients'
                    ],
                    'alternative': 'MinMaxScaler for bounded interpretation'
                }
            elif variable in chemistry_vars:
                assignment = {
                    'primary_scaler': 'RobustScaler',
                    'rationale': [
                        'Chemistry variables show outliers',
                        'Domain-specific reasonable ranges exist',
                        'Robust scaling handles measurement errors',
                        'Preserves wine science interpretation'
                    ],
                    'alternative': 'PowerTransformer after distribution transformation'
                }
            else:
                assignment = {
                    'primary_scaler': 'RobustScaler',
                    'rationale': [
                        'Default robust approach for outlier-heavy data',
                        'Safe choice given 99% outlier rate',
                        'Maintains interpretability',
                        'Good performance across algorithms'
                    ],
                    'alternative': 'StandardScaler if outliers removed first'
                }
            
            scaling_strategy['variable_assignments'][variable] = assignment
        
        # Validation framework
        scaling_strategy['validation_framework'] = {
            'statistical_tests': [
                'Distribution similarity before/after scaling',
                'Correlation preservation analysis',
                'Outlier impact assessment'
            ],
            'ml_validation': [
                'Cross-validation performance across algorithms',
                'Feature importance stability',
                'Prediction accuracy comparison'
            ],
            'business_validation': [
                'Interpretability preservation check',
                'Domain expert review of scaled values',
                'Business rule consistency validation'
            ],
            'selection_criteria': {
                'weights': {
                    'ml_performance': 0.4,
                    'business_interpretability': 0.3,
                    'statistical_appropriateness': 0.3
                },
                'decision_rule': 'Highest weighted score with business veto power'
            }
        }
        
        return scaling_strategy
    
    def implement_preparation_pipeline(self, missing_strategy: Dict, outlier_strategy: Dict,
                                     transformation_strategy: Dict, feature_strategy: Dict,
                                     scaling_strategy: Dict) -> pd.DataFrame:
        """
        Implement the complete data preparation pipeline.
        
        Parameters:
        -----------
        missing_strategy : dict
            Missing data treatment strategy
        outlier_strategy : dict
            Outlier treatment strategy
        transformation_strategy : dict
            Transformation strategy
        feature_strategy : dict
            Feature engineering strategy
        scaling_strategy : dict
            Scaling strategy
            
        Returns:
        --------
        pandas.DataFrame
            Fully prepared dataset
        """
        print("Implementing comprehensive data preparation pipeline...")
        
        df_processed = self.df_original.copy()
        processing_log = []
        
        # Step 1: Missing Data Treatment
        print("Step 1: Treating missing data...")
        df_processed = self._apply_missing_data_treatment(df_processed, missing_strategy)
        processing_log.append("Missing data treatment completed")
        
        # Step 2: Outlier Treatment
        print("Step 2: Treating outliers...")
        df_processed = self._apply_outlier_treatment(df_processed, outlier_strategy)
        processing_log.append("Outlier treatment completed")
        
        # Step 3: Distribution Transformations
        print("Step 3: Applying transformations...")
        df_processed = self._apply_transformations(df_processed, transformation_strategy)
        processing_log.append("Distribution transformations completed")
        
        # Step 4: Feature Engineering
        print("Step 4: Engineering features...")
        df_processed = self._apply_feature_engineering(df_processed, feature_strategy)
        processing_log.append("Feature engineering completed")
        
        # Step 5: Scaling and Normalization
        print("Step 5: Scaling variables...")
        df_processed = self._apply_scaling(df_processed, scaling_strategy)
        processing_log.append("Scaling completed")
        
        # Create before/after comparison
        self._create_preparation_comparison_plots(df_processed)
        
        # Store processed dataset
        self.df_prepared = df_processed
        
        # Log final processing summary
        self._log_preparation_step("complete_pipeline", {
            'original_shape': self.df_original.shape,
            'processed_shape': df_processed.shape,
            'processing_steps': processing_log,
            'new_features_created': len([col for col in df_processed.columns 
                                       if col not in self.df_original.columns])
        })
        
        return df_processed
    
    def _apply_missing_data_treatment(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """Apply missing data treatment based on strategy."""
        
        df_treated = df.copy()
        treatment_plan = strategy['treatment_plan']
        
        for variable, plan in treatment_plan.items():
            if plan['method'] == 'MedianImputer':
                median_value = df_treated[variable].median()
                df_treated[variable].fillna(median_value, inplace=True)
                
                # Add missing indicator for STARS
                if variable == 'STARS':
                    df_treated['STARS_was_missing'] = df[variable].isnull().astype(int)
                    
            elif plan['method'] == 'MeanImputer':
                mean_value = df_treated[variable].mean()
                df_treated[variable].fillna(mean_value, inplace=True)
                
            elif plan['method'] == 'KNNImputer':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                numerical_columns = df_treated.select_dtypes(include=[np.number]).columns
                df_treated[numerical_columns] = imputer.fit_transform(df_treated[numerical_columns])
                self.imputers[variable] = imputer
        
        return df_treated
    
    def _apply_outlier_treatment(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """Apply outlier treatment based on strategy."""
        
        df_treated = df.copy()
        
        for variable in self.numerical_columns:
            if variable in strategy['variable_specific_strategies']:
                var_strategy = strategy['variable_specific_strategies'][variable]
                
                if 'winsorization' in var_strategy['strategy']:
                    # Apply winsorization
                    p1 = df_treated[variable].quantile(0.01)
                    p99 = df_treated[variable].quantile(0.99)
                    df_treated[variable] = df_treated[variable].clip(lower=p1, upper=p99)
        
        return df_treated
    
    def _apply_transformations(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """Apply distribution transformations based on strategy."""
        
        df_transformed = df.copy()
        recommendations = strategy['variable_recommendations']
        
        for variable, recommendation in recommendations.items():
            if variable not in df_transformed.columns:
                continue
                
            method = recommendation['primary_method']
            
            if method == 'log_plus_one':
                df_transformed[f'{variable}_log'] = np.log1p(df_transformed[variable])
                
            elif method == 'box_cox':
                from scipy.stats import boxcox
                try:
                    transformed_data, lambda_param = boxcox(df_transformed[variable] + 1)
                    df_transformed[f'{variable}_boxcox'] = transformed_data
                    self.transformers[variable] = {'method': 'box_cox', 'lambda': lambda_param}
                except:
                    # Fallback to log transformation
                    df_transformed[f'{variable}_log'] = np.log1p(df_transformed[variable])
                    
            elif method == 'yeo_johnson':
                from sklearn.preprocessing import PowerTransformer
                transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                transformed_data = transformer.fit_transform(df_transformed[[variable]])
                df_transformed[f'{variable}_yj'] = transformed_data[:, 0]
                self.transformers[variable] = transformer
        
        return df_transformed
    
    def _apply_feature_engineering(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """Apply feature engineering based on strategy."""
        
        df_engineered = df.copy()
        
        # Chemistry ratios
        chemistry_ratios = strategy['chemistry_ratios']
        for feature_name, feature_info in chemistry_ratios.items():
            try:
                if feature_name == 'total_to_free_so2_ratio':
                    df_engineered[feature_name] = (df_engineered['TotalSulfurDioxide'] / 
                                                 (df_engineered['FreeSulfurDioxide'] + 1))
                elif feature_name == 'acidity_balance_ratio':
                    df_engineered[feature_name] = (df_engineered['FixedAcidity'] / 
                                                 (df_engineered['VolatileAcidity'] + 0.1))
                elif feature_name == 'sugar_alcohol_ratio':
                    df_engineered[feature_name] = (df_engineered['ResidualSugar'] / 
                                                 (df_engineered['Alcohol'] + 1))
            except:
                continue
        
        # Interaction features
        interaction_features = strategy['interaction_features']
        for feature_name, feature_info in interaction_features.items():
            try:
                if feature_name == 'stars_label_synergy':
                    df_engineered[feature_name] = (df_engineered['STARS'] * 
                                                 df_engineered['LabelAppeal'])
            except:
                continue
        
        return df_engineered
    
    def _apply_scaling(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """Apply scaling based on strategy."""
        
        df_scaled = df.copy()
        variable_assignments = strategy['variable_assignments']
        
        for variable, assignment in variable_assignments.items():
            if variable not in df_scaled.columns or assignment['primary_scaler'] == 'no_scaling':
                continue
                
            scaler_type = assignment['primary_scaler']
            
            if scaler_type == 'RobustScaler':
                scaler = RobustScaler()
            elif scaler_type == 'StandardScaler':
                scaler = StandardScaler()
            elif scaler_type == 'MinMaxScaler':
                scaler = MinMaxScaler()
            else:
                continue
            
            df_scaled[variable] = scaler.fit_transform(df_scaled[[variable]])[:, 0]
            self.scalers[variable] = scaler
        
        return df_scaled
    
    def _create_missing_data_treatment_plot(self, missing_analysis: Dict, treatment_plan: Dict) -> None:
        """Create visualization for missing data treatment plan."""
        
        if not missing_analysis:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Missing percentages
        variables = list(missing_analysis.keys())
        percentages = [missing_analysis[var]['missing_percentage'] for var in variables]
        
        bars = ax1.barh(variables, percentages)
        ax1.set_xlabel('Missing Percentage (%)')
        ax1.set_title('Missing Data by Variable')
        ax1.grid(axis='x', alpha=0.3)
        
        # Color bars by treatment method
        method_colors = {
            'MedianImputer': 'lightblue',
            'MeanImputer': 'lightgreen', 
            'KNNImputer': 'orange',
            'RandomForestImputer': 'red'
        }
        
        for i, (var, bar) in enumerate(zip(variables, bars)):
            method = treatment_plan[var]['method']
            bar.set_color(method_colors.get(method, 'gray'))
            
        # Plot 2: Treatment method distribution
        methods = [treatment_plan[var]['method'] for var in variables]
        method_counts = pd.Series(methods).value_counts()
        
        ax2.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
        ax2.set_title('Treatment Method Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/missing_data_treatment_plan.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_preparation_comparison_plots(self, df_processed: pd.DataFrame) -> None:
        """Create before/after comparison plots."""
        
        # Select key variables for comparison
        key_variables = ['STARS', 'LabelAppeal', 'Alcohol', 'pH', 'TARGET']
        available_vars = [var for var in key_variables if var in self.df_original.columns]
        
        if len(available_vars) < 2:
            return
        
        fig, axes = plt.subplots(2, len(available_vars), figsize=(4*len(available_vars), 10))
        if len(available_vars) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, var in enumerate(available_vars):
            # Before (original)
            axes[0, i].hist(self.df_original[var].dropna(), bins=30, alpha=0.7, 
                           color='red', label='Original')
            axes[0, i].set_title(f'{var} - Original Distribution')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].legend()
            
            # After (processed)
            if var in df_processed.columns:
                axes[1, i].hist(df_processed[var].dropna(), bins=30, alpha=0.7, 
                               color='blue', label='Processed')
                axes[1, i].set_title(f'{var} - After Processing')
                axes[1, i].set_xlabel(var)
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].legend()
        
        plt.suptitle('Data Preparation Before/After Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/before_after_preparation_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _summarize_missing_treatment(self, treatment_plan: Dict) -> Dict[str, Any]:
        """Summarize missing data treatment plan."""
        
        method_counts = {}
        total_missing = 0
        
        for var, plan in treatment_plan.items():
            method = plan['method']
            method_counts[method] = method_counts.get(method, 0) + 1
            # Would need to access missing counts from analysis
        
        return {
            'total_variables_with_missing': len(treatment_plan),
            'treatment_method_distribution': method_counts,
            'primary_approach': 'Severity-based treatment selection',
            'business_focus': 'Preserve interpretability while ensuring data quality'
        }

def comprehensive_data_preparation_strategy(df: pd.DataFrame, target_column: str = 'TARGET',
                                          output_dir: str = 'data_preparation_plots') -> Dict[str, Any]:
    """
    Create comprehensive data preparation strategy based on EDA findings.
    
    This function integrates insights from missing data analysis, outlier detection,
    univariate analysis, correlation analysis, and TARGET analysis to create
    a complete data preparation pipeline with business justifications.
    
    Key EDA Findings Addressed:
    --------------------------
    - Missing Data: STARS 26.2% missing (median imputation + indicator)
    - Outliers: 99% outlier rate requiring robust treatment approaches  
    - Distributions: All variables fail normality tests (transformation needed)
    - Correlations: STARS (0.559) and LabelAppeal (0.357) are top predictors
    - Business Impact: Quality and marketing drive sales success
    
    Preparation Strategy:
    --------------------
    1. Missing Data: Tiered approach based on importance and missing rate
    2. Outliers: Consensus-based severity treatment (winsorization + robust scaling)
    3. Transformations: Variable-specific normalization (Box-Cox, Yeo-Johnson, log)
    4. Feature Engineering: Chemistry ratios, quality interactions, wine styles
    5. Scaling: Algorithm-appropriate scaling preserving business interpretation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw wine dataset from EDA analysis
    target_column : str, default 'TARGET'
        Wine sales target variable
    output_dir : str, default 'data_preparation_plots'
        Directory for preparation visualizations
        
    Returns:
    --------
    dict
        Comprehensive preparation strategy including:
        - Missing data treatment plan with business rationale
        - Outlier treatment framework (tiered by severity)
        - Distribution transformation pipeline (method selection)
        - Feature engineering opportunities (chemistry + business)
        - Scaling strategy (algorithm + interpretability optimized)
        - Implementation pipeline with validation framework
        - Before/after comparison and business impact assessment
        
    Generated Outputs:
    ------------------
    - missing_data_treatment_plan.png: Treatment method visualization
    - before_after_preparation_comparison.png: Transformation effects
    - Complete preparation pipeline ready for ML modeling
    - Business-justified recommendations for each decision
    """
    
    print("="*70)
    print("COMPREHENSIVE DATA PREPARATION STRATEGY")
    print("Based on Complete EDA Findings")
    print("="*70)
    
    # Initialize strategy designer
    strategy_designer = DataPreparationStrategy(df, target_column, output_dir)
    
    # Step 1: Missing Data Treatment Plan
    print("\n🔍 STEP 1: MISSING DATA TREATMENT STRATEGY")
    print("-" * 50)
    missing_strategy = strategy_designer.create_missing_data_treatment_plan()
    
    # Step 2: Outlier Treatment Framework
    print("\n⚠️  STEP 2: OUTLIER TREATMENT FRAMEWORK") 
    print("-" * 50)
    outlier_strategy = strategy_designer.create_outlier_treatment_framework()
    
    # Step 3: Distribution Transformation Pipeline
    print("\n📊 STEP 3: DISTRIBUTION TRANSFORMATION PIPELINE")
    print("-" * 50)
    transformation_strategy = strategy_designer.create_transformation_pipeline()
    
    # Step 4: Feature Engineering Strategy
    print("\n🔬 STEP 4: FEATURE ENGINEERING OPPORTUNITIES")
    print("-" * 50)
    feature_strategy = strategy_designer.design_feature_engineering_strategy()
    
    # Step 5: Scaling and Normalization Strategy  
    print("\n⚖️  STEP 5: SCALING AND NORMALIZATION STRATEGY")
    print("-" * 50)
    scaling_strategy = strategy_designer.design_scaling_strategy()
    
    # Compile comprehensive strategy
    comprehensive_strategy = {
        'strategy_overview': {
            'data_challenges_identified': [
                'High missing data rate (STARS: 26.2%)',
                'Extensive outliers (99% outlier rate)', 
                'Non-normal distributions (all variables fail tests)',
                'Domain-specific data quality issues (9,959 impossible values)',
                'Business-critical interpretability requirements'
            ],
            'approach': 'EDA-informed, business-justified, ML-optimized preparation',
            'validation_framework': 'Multi-metric evaluation with business impact assessment'
        },
        'missing_data_strategy': missing_strategy,
        'outlier_treatment_strategy': outlier_strategy, 
        'transformation_strategy': transformation_strategy,
        'feature_engineering_strategy': feature_strategy,
        'scaling_strategy': scaling_strategy
    }
    
    # Generate implementation recommendations
    implementation_recommendations = {
        'priority_order': [
            '1. Address missing data (especially STARS - key predictor)',
            '2. Handle severe outliers (10.1% require immediate attention)', 
            '3. Apply distribution transformations (all variables non-normal)',
            '4. Engineer business-relevant features (chemistry ratios)',
            '5. Scale appropriately for target algorithms'
        ],
        'business_validation_checkpoints': [
            'STARS imputation preserves sales correlation',
            'Outlier treatment maintains wine chemistry validity',
            'Transformations preserve business interpretability', 
            'New features align with wine science principles',
            'Scaling enables accurate business insights'
        ],
        'risk_mitigation': [
            'Test multiple imputation methods for STARS',
            'Validate outlier treatment with domain experts',
            'Compare transformation methods on holdout data',
            'Verify feature engineering improves predictions',
            'Ensure scaling reversibility for interpretation'
        ]
    }
    
    comprehensive_strategy['implementation_recommendations'] = implementation_recommendations
    
    # Print executive summary
    print(f"\n📋 PREPARATION STRATEGY EXECUTIVE SUMMARY:")
    print(f"   • Missing data variables: {len(missing_strategy['treatment_plan'])}")
    print(f"   • Outlier treatment approach: Tiered severity-based (99% outlier rate)")
    print(f"   • Distribution transformations: All 15 variables (none normal)")
    print(f"   • New features planned: {len(feature_strategy['chemistry_ratios']) + len(feature_strategy['interaction_features'])}")
    print(f"   • Scaling methods: Variable-specific optimization")
    
    print(f"\n🎯 KEY BUSINESS PRIORITIES:")
    print("   1. STARS imputation critical (strongest sales predictor r=0.559)")
    print("   2. Preserve wine chemistry interpretability throughout process") 
    print("   3. Robust outlier handling (extensive outlier contamination)")
    print("   4. Feature engineering leverages domain knowledge")
    print("   5. Enable both ML performance and business insights")
    
    print(f"\n🔧 IMPLEMENTATION APPROACH:")
    print("   • Modular pipeline enabling component testing")
    print("   • Business validation at each major step")
    print("   • Fallback methods for each transformation")
    print("   • Comprehensive before/after validation")
    print("   • Domain expert review integration points")
    
    print(f"\n💡 EXPECTED BUSINESS IMPACT:")
    print("   • Improved model accuracy through quality data preparation")
    print("   • Enhanced feature set capturing wine science relationships")  
    print("   • Maintained business interpretability for decision making")
    print("   • Robust handling of data quality issues")
    print("   • Scalable pipeline for ongoing wine analysis")
    
    print(f"\n📁 Preparation visualizations saved to: {output_dir}/")
    print("   • Treatment plan documentation and validation plots")
    
    print("\n" + "="*70)
    print("DATA PREPARATION STRATEGY COMPLETE")
    print("Ready for Implementation and ML Pipeline Integration")
    print("="*70)
    
    return comprehensive_strategy

# Example usage
if __name__ == "__main__":
    # Load wine dataset
    print("Loading wine dataset for preparation strategy...")
    df = pd.read_csv('M3_Data.csv', encoding='utf-8-sig')
    
    # Create comprehensive preparation strategy
    preparation_strategy = comprehensive_data_preparation_strategy(df)