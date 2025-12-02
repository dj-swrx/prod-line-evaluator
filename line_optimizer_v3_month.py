# Enhanced Analysis Plan with Temporal Data
# With 30 days of data (30 datasets × 40 production lines = 1200 samples total), 
# we can perform much more robust analysis.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TemporalProductionOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.monthly_data = {}
        self.consolidated_data = None
        self.feature_importance_monthly = {}
        
    def load_monthly_data(self, daily_datasets):
        """Load and consolidate 30 days of production data"""
        self.daily_datasets = daily_datasets
        self.n_days = len(daily_datasets)
        
        # Consolidate all data
        all_data = []
        for day, (X_day, y_day) in daily_datasets.items():
            day_data = X_day.copy()
            day_data['error_percentage'] = y_day
            day_data['day'] = day
            day_data['line_id'] = range(len(X_day))
            all_data.append(day_data)
        
        self.consolidated_data = pd.concat(all_data, ignore_index=True)
        self.X_consolidated = self.consolidated_data.drop(['error_percentage', 'day', 'line_id'], axis=1)
        self.y_consolidated = self.consolidated_data['error_percentage']
        
        print(f"Loaded {self.n_days} days of data")
        print(f"Total samples: {len(self.consolidated_data)}")
        print(f"Total parameters: {self.X_consolidated.shape[1]}")
        
    def temporal_feature_analysis(self):
        """Comprehensive temporal feature importance analysis"""
        
        print("🔍 PERFORMING TEMPORAL FEATURE ANALYSIS")
        print("=" * 50)
        
        results = {
            'daily_analysis': self._daily_consistency_analysis(),
            'stability_analysis': self._feature_stability_analysis(),
            'temporal_trends': self._temporal_trend_analysis(),
            'consolidated_importance': self._consolidated_feature_importance()
        }
        
        return results
    
    def _daily_consistency_analysis(self):
        """Analyze feature importance consistency across days"""
        daily_importances = {}
        
        for day, (X_day, y_day) in self.daily_datasets.items():
            # Random Forest importance for each day
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_day, y_day)
            daily_importances[day] = pd.Series(rf.feature_importances_, 
                                             index=X_day.columns).sort_values(ascending=False)
        
        # Calculate consistency scores
        consistency_df = self._calculate_consistency_scores(daily_importances)
        return consistency_df
    
    def _calculate_consistency_scores(self, daily_importances):
        """Calculate how consistently important each feature is across days"""
        all_features = list(daily_importances.values())[0].index
        
        consistency_data = []
        for feature in all_features:
            daily_ranks = []
            daily_scores = []
            
            for day, importance_series in daily_importances.items():
                if feature in importance_series:
                    rank = importance_series.index.get_loc(feature) + 1  # 1-based ranking
                    score = importance_series[feature]
                    daily_ranks.append(rank)
                    daily_scores.append(score)
            
            if daily_ranks:
                consistency_data.append({
                    'feature': feature,
                    'mean_rank': np.mean(daily_ranks),
                    'rank_std': np.std(daily_ranks),
                    'mean_importance': np.mean(daily_scores),
                    'importance_std': np.std(daily_scores),
                    'rank_stability': 1 / (1 + np.std(daily_ranks)),  # Higher = more stable
                    'appearance_frequency': len(daily_ranks) / len(daily_importances)
                })
        
        consistency_df = pd.DataFrame(consistency_data)
        consistency_df['consistency_score'] = (
            consistency_df['rank_stability'] * 
            consistency_df['appearance_frequency'] * 
            consistency_df['mean_importance']
        )
        
        return consistency_df.sort_values('consistency_score', ascending=False)
    
    def _feature_stability_analysis(self):
        """Analyze parameter stability over time"""
        stability_data = []
        
        for feature in self.X_consolidated.columns:
            feature_stability = self._analyze_parameter_stability(feature)
            stability_data.append(feature_stability)
        
        stability_df = pd.DataFrame(stability_data)
        return stability_df.sort_values('variation_coefficient')
    
    def _analyze_parameter_stability(self, feature):
        """Analyze how stable a parameter is over time"""
        daily_values = []
        daily_impacts = []
        
        for day, (X_day, y_day) in self.daily_datasets.items():
            if feature in X_day.columns:
                daily_values.extend(X_day[feature].values)
                
                # Simple correlation for impact
                correlation = X_day[feature].corr(y_day)
                daily_impacts.append(abs(correlation))
        
        values_array = np.array(daily_values)
        impacts_array = np.array(daily_impacts)
        
        return {
            'feature': feature,
            'mean_value': np.mean(values_array),
            'std_value': np.std(values_array),
            'variation_coefficient': np.std(values_array) / np.mean(values_array),
            'mean_impact': np.mean(impacts_array),
            'impact_consistency': 1 - np.std(impacts_array) / (np.mean(impacts_array) + 1e-8)
        }
    
    def _temporal_trend_analysis(self):
        """Analyze how feature importance changes over time"""
        temporal_trends = {}
        
        # Calculate rolling importance
        window_size = 7  # Weekly windows
        for feature in self.X_consolidated.columns[:100]:  # Analyze top 100 for efficiency
            trend = self._calculate_feature_trend(feature, window_size)
            temporal_trends[feature] = trend
        
        return temporal_trends
    
    def _calculate_feature_trend(self, feature, window_size):
        """Calculate temporal trend for a feature"""
        daily_correlations = []
        days = sorted(self.daily_datasets.keys())
        
        for day in days:
            X_day, y_day = self.daily_datasets[day]
            if feature in X_day.columns:
                corr = abs(X_day[feature].corr(y_day))
                daily_correlations.append(corr)
            else:
                daily_correlations.append(np.nan)
        
        # Calculate trend
        valid_correlations = [c for c in daily_correlations if not np.isnan(c)]
        if len(valid_correlations) >= 2:
            slope, _, _, _, _ = stats.linregress(range(len(valid_correlations)), valid_correlations)
            trend_strength = abs(slope)
        else:
            trend_strength = 0
        
        return {
            'daily_correlations': daily_correlations,
            'trend_strength': trend_strength,
            'mean_correlation': np.mean(valid_correlations)
        }
    
    def _consolidated_feature_importance(self):
        """Calculate feature importance on consolidated monthly data"""
        print("Calculating consolidated feature importance...")
        
        X_scaled = self.scaler.fit_transform(self.X_consolidated)
        
        # Multiple feature importance methods on full dataset
        importance_methods = {}
        
        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, self.y_consolidated)
        importance_methods['random_forest'] = pd.Series(rf.feature_importances_, 
                                                       index=self.X_consolidated.columns)
        
        # 2. Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
        gb.fit(X_scaled, self.y_consolidated)
        importance_methods['gradient_boosting'] = pd.Series(gb.feature_importances_, 
                                                          index=self.X_consolidated.columns)
        
        # 3. Lasso
        lasso = Lasso(alpha=0.001, random_state=42, max_iter=5000)
        lasso.fit(X_scaled, self.y_consolidated)
        importance_methods['lasso'] = pd.Series(np.abs(lasso.coef_), 
                                              index=self.X_consolidated.columns)
        
        # 4. Mutual Information
        mi_scores = mutual_info_regression(X_scaled, self.y_consolidated, 
                                         random_state=42, n_neighbors=5)
        importance_methods['mutual_info'] = pd.Series(mi_scores, 
                                                    index=self.X_consolidated.columns)
        
        # Combine all methods
        importance_df = pd.DataFrame(importance_methods)
        importance_df = importance_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        importance_df['consolidated_score'] = importance_df.mean(axis=1)
        
        return importance_df.sort_values('consolidated_score', ascending=False)
    
    def identify_most_influential_parameters(self, top_k=20):
        """Identify the most influential parameters considering temporal stability"""
        
        print("\n🎯 IDENTIFYING MOST INFLUENTIAL PARAMETERS")
        print("=" * 50)
        
        # Get all analysis results
        temporal_results = self.temporal_feature_analysis()
        
        # Combine scores from different analyses
        consolidated_importance = temporal_results['consolidated_importance']
        consistency_scores = temporal_results['daily_analysis']
        stability_analysis = temporal_results['stability_analysis']
        
        # Create comprehensive ranking
        comprehensive_ranking = []
        
        for feature in consolidated_importance.index:
            if feature in consistency_scores['feature'].values and feature in stability_analysis['feature'].values:
                cons_score = consolidated_importance.loc[feature, 'consolidated_score']
                consistency_row = consistency_scores[consistency_scores['feature'] == feature].iloc[0]
                stability_row = stability_analysis[stability_analysis['feature'] == feature].iloc[0]
                
                # Combined score considering importance, consistency, and stability
                combined_score = (
                    cons_score * 0.5 +  # Raw importance
                    consistency_row['consistency_score'] * 0.3 +  # Daily consistency
                    stability_row['impact_consistency'] * 0.2  # Impact stability
                )
                
                comprehensive_ranking.append({
                    'feature': feature,
                    'combined_score': combined_score,
                    'raw_importance': cons_score,
                    'daily_consistency': consistency_row['consistency_score'],
                    'impact_stability': stability_row['impact_consistency'],
                    'parameter_stability': 1 - stability_row['variation_coefficient'],
                    'mean_rank': consistency_row['mean_rank']
                })
        
        ranking_df = pd.DataFrame(comprehensive_ranking)
        ranking_df = ranking_df.sort_values('combined_score', ascending=False)
        
        # Display top parameters
        print(f"\nTop {top_k} Most Influential Parameters (Considering Temporal Stability):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Parameter':<20} {'Combined Score':<15} {'Raw Imp':<10} {'Consistency':<12} {'Stability':<10}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(ranking_df.head(top_k).iterrows(), 1):
            print(f"{i:<4} {row['feature']:<20} {row['combined_score']:<15.4f} "
                  f"{row['raw_importance']:<10.4f} {row['daily_consistency']:<12.4f} "
                  f"{row['impact_stability']:<10.4f}")
        
        return ranking_df.head(top_k)
    
    def generate_temporal_visualizations(self, top_parameters=15):
        """Generate comprehensive temporal visualizations"""
        
        temporal_results = self.temporal_feature_analysis()
        top_params = self.identify_most_influential_parameters(top_parameters)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Feature Importance Analysis - 30 Days Data', fontsize=16, fontweight='bold')
        
        # 1. Top parameters by combined score
        top_params.head(15).plot(x='feature', y='combined_score', kind='barh', ax=axes[0,0])
        axes[0,0].set_title('Top Parameters - Combined Importance Score')
        axes[0,0].set_xlabel('Combined Score')
        
        # 2. Consistency vs Importance
        scatter_data = temporal_results['daily_analysis'].head(50)
        axes[0,1].scatter(scatter_data['mean_importance'], scatter_data['rank_stability'], alpha=0.6)
        axes[0,1].set_xlabel('Mean Importance')
        axes[0,1].set_ylabel('Rank Stability')
        axes[0,1].set_title('Importance vs Stability')
        
        # Add labels for top points
        for i, (_, row) in enumerate(scatter_data.head(10).iterrows()):
            axes[0,1].annotate(row['feature'], (row['mean_importance'], row['rank_stability']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Parameter stability distribution
        stability_data = temporal_results['stability_analysis']
        axes[1,0].hist(stability_data['variation_coefficient'], bins=30, alpha=0.7)
        axes[1,0].axvline(stability_data['variation_coefficient'].median(), color='red', linestyle='--')
        axes[1,0].set_xlabel('Coefficient of Variation')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Parameter Value Stability Distribution')
        
        # 4. Daily appearance frequency
        consistency_data = temporal_results['daily_analysis']
        axes[1,1].hist(consistency_data['appearance_frequency'], bins=20, alpha=0.7)
        axes[1,1].set_xlabel('Daily Appearance Frequency')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('How Often Parameters Appear as Important')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage with synthetic monthly data
def create_monthly_production_data():
    """Create 30 days of synthetic production data"""
    monthly_data = {}
    
    for day in range(1, 31):
        np.random.seed(42 + day)  # Different seed each day
        
        n_lines = 40
        n_params = 2500
        
        X_day = pd.DataFrame(
            np.random.normal(100, 10, (n_lines, n_params)),
            columns=[f'param_{i:04d}' for i in range(n_params)]
        )
        
        # Consistent important parameters across month
        consistently_important = {
            'param_0042': 2.5,   # Always important
            'param_0127': -1.8,  # Always important  
            'param_0456': 3.2,   # Always important
        }
        
        # Some parameters that vary in importance
        varying_important = {
            'param_0789': -2.1 * (0.8 + 0.4 * np.sin(day/5)),  # Varies
            'param_1234': 1.5 * (0.7 + 0.6 * np.cos(day/7)),   # Varies
        }
        
        # Generate error
        y_day = 5.0  # Base error
        for param, coef in consistently_important.items():
            y_day += coef * (X_day[param] - X_day[param].mean()) / X_day[param].std()
        
        for param, coef in varying_important.items():
            y_day += coef * (X_day[param] - X_day[param].mean()) / X_day[param].std()
        
        # Add some noise and daily variations
        y_day += np.random.normal(0, 1.0, n_lines)
        y_day += np.sin(day/3) * 0.5  # Weekly pattern
        
        monthly_data[day] = (X_day, y_day)
    
    print(f"Created {len(monthly_data)} days of production data")
    return monthly_data

def main():
    """Main analysis with monthly data"""
    print("🏭 MONTHLY PRODUCTION DATA ANALYSIS")
    print("=" * 50)
    
    # Create monthly data
    monthly_data = create_monthly_production_data()
    
    # Initialize temporal analyzer
    analyzer = TemporalProductionOptimizer()
    analyzer.load_monthly_data(monthly_data)
    
    # Identify most influential parameters
    top_parameters = analyzer.identify_most_influential_parameters(top_k=20)
    
    # Generate visualizations
    print("\n📊 GENERATING TEMPORAL ANALYSIS VISUALIZATIONS...")
    analyzer.generate_temporal_visualizations()
    
    print("\n✅ Monthly analysis completed!")
    
    return analyzer, top_parameters

if __name__ == "__main__":
    analyzer, top_parameters = main()