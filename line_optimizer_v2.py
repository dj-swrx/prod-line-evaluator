import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ProductionOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.optimization_model = None
        self.top_features = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.current_predictions = None
        
    def load_data(self, X, y):
        """Utility: Load and validate data"""
        self.X = X
        self.y = y
        self.feature_names = self.X.columns.tolist()
        print(f"Data loaded: {self.X.shape[0]} lines, {self.X.shape[1]} parameters")
        
    def analyze_feature_importance(self):
        """Utility: Comprehensive feature importance analysis"""
        if self.X is None:
            raise ValueError("No data loaded. Please call load_data() first.")
            
        # Multiple feature importance methods
        results = {}
        
        # 1. Correlation with target
        results['correlation'] = self.X.corrwith(self.y).abs()
        
        # 2. Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        results['random_forest'] = pd.Series(rf.feature_importances_, index=self.feature_names)
        
        # 3. Lasso importance
        X_scaled = self.scaler.fit_transform(self.X)
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_scaled, self.y)
        results['lasso'] = pd.Series(np.abs(lasso.coef_), index=self.feature_names)
        
        # Combine scores
        importance_df = pd.DataFrame(results)
        importance_df['combined_score'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df

    def visualization_utilities(self, top_k=15):
        """Utility: Visualization functions with better display handling"""
        if self.feature_importance is None:
            self.analyze_feature_importance()
            
        top_features = self.feature_importance.head(top_k)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Production Line Optimization Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top feature importance (horizontal bar chart)
        y_pos = np.arange(len(top_features))
        axes[0,0].barh(y_pos, top_features['combined_score'])
        axes[0,0].set_yticks(y_pos)
        axes[0,0].set_yticklabels(top_features.index)
        axes[0,0].set_xlabel('Importance Score')
        axes[0,0].set_title('Top Feature Importance Scores')
        axes[0,0].grid(axis='x', alpha=0.3)
        
        # 2. Feature importance comparison
        top_features[['correlation', 'random_forest', 'lasso']].plot(
            kind='bar', ax=axes[0,1], color=['#ff9999', '#66b3ff', '#99ff99']
        )
        axes[0,1].set_title('Feature Importance by Method')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(loc='upper right')
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # 3. Current vs predicted error
        if hasattr(self, 'current_predictions') and self.current_predictions is not None:
            axes[1,0].scatter(self.y, self.current_predictions, alpha=0.7, color='blue', s=50)
            axes[1,0].plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', lw=2, label='Perfect prediction')
            axes[1,0].set_xlabel('Actual Error (%)')
            axes[1,0].set_ylabel('Predicted Error (%)')
            axes[1,0].set_title('Model Prediction vs Actual')
            axes[1,0].legend()
            axes[1,0].grid(alpha=0.3)
            
            # Add R² score to plot
            from sklearn.metrics import r2_score
            r2 = r2_score(self.y, self.current_predictions)
            axes[1,0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1,0].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            axes[1,0].text(0.5, 0.5, 'Run optimization first\nto see predictions', 
                        ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
            axes[1,0].set_title('Model Predictions (Run optimization first)')
        
        # 4. Error distribution
        n, bins, patches = axes[1,1].hist(self.y, bins=12, alpha=0.7, edgecolor='black', color='lightblue')
        axes[1,1].axvline(self.y.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {self.y.mean():.2f}%')
        axes[1,1].axvline(self.y.median(), color='orange', linestyle='--', lw=2, label=f'Median: {self.y.median():.2f}%')
        axes[1,1].set_xlabel('Error Percentage (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Error Distribution Across Production Lines')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Show the plot
        print("Displaying visualizations... Close the plot window to continue.")
        plt.show(block=True)

    def optimization_utilities(self, top_k=10):
        """Utility: Build optimization model and find optimal settings"""
        if self.feature_importance is None:
            self.analyze_feature_importance()
            
        self.top_features = self.feature_importance.head(top_k).index.tolist()
        
        print(f"Selected top {len(self.top_features)} features for optimization:")
        for i, feature in enumerate(self.top_features, 1):
            score = self.feature_importance.loc[feature, 'combined_score']
            print(f"  {i}. {feature} (score: {score:.4f})")
        
        # Train model on top features with better parameters
        self.optimization_model = GradientBoostingRegressor(
            n_estimators=200, 
            random_state=42,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=3
        )
        self.optimization_model.fit(self.X[self.top_features], self.y)
        
        # Store predictions for analysis
        self.current_predictions = self.optimization_model.predict(self.X[self.top_features])
        
        # Calculate model performance
        train_score = self.optimization_model.score(self.X[self.top_features], self.y)
        print(f"Model R² score: {train_score:.4f}")
        
        return self.top_features

    def optimization_objective(self, params):
        """Utility: Objective function for optimization (minimize error)"""
        if self.optimization_model is None:
            raise ValueError("Optimization model not trained. Call optimization_utilities() first.")
        
        params_2d = np.array(params).reshape(1, -1)
        predicted_error = self.optimization_model.predict(params_2d)[0]
        
        # Add small penalty for large changes from median to encourage meaningful but reasonable adjustments
        medians = self.X[self.top_features].median().values
        change_penalty = 0.001 * np.sum((params - medians) ** 2)
        
        return predicted_error + change_penalty

    def constraint_utilities(self):
        """Utility: Define constraints for optimization"""
        if self.top_features is None:
            raise ValueError("Top features not selected. Call optimization_utilities() first.")
            
        bounds = []
        for feature in self.top_features:
            current_min = self.X[feature].min()
            current_max = self.X[feature].max()
            current_std = self.X[feature].std()
            
            # More realistic bounds that allow meaningful changes
            # Allow parameters to vary within ±3 standard deviations or actual range
            lower_bound = max(current_min - current_std, current_min * 0.7)  # Don't go too low
            upper_bound = min(current_max + current_std, current_max * 1.3)  # Don't go too high
            
            bounds.append((lower_bound, upper_bound))
            
        return bounds

    def find_optimal_parameters(self, top_k=10):
        """Main optimization function using utility functions"""
        if not hasattr(self, 'optimization_model') or self.optimization_model is None:
            self.optimization_utilities(top_k=top_k)
        
        if self.top_features is None:
            raise ValueError("Top features not initialized properly.")
            
        # Use median current settings as starting point
        initial_params = self.X[self.top_features].median().values
        
        # Get bounds from constraint utility
        bounds = self.constraint_utilities()
        
        # Try multiple starting points to avoid local minima
        best_result = None
        best_error = float('inf')
        
        print("Running optimization with multiple starting points...")
        
        for attempt in range(5):
            if attempt > 0:
                # Add some randomness to initial parameters
                initial_params = self.X[self.top_features].sample(1).values[0]
            
            result = minimize(
                self.optimization_objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_result = result
                print(f"  Attempt {attempt + 1}: Found solution with error {result.fun:.4f}")
        
        if best_result is None:
            print("All optimization attempts failed. Using first result.")
            best_result = result
        
        if best_result.success:
            optimal_settings = dict(zip(self.top_features, best_result.x))
            min_error = best_result.fun
            
            print("\n🎯 OPTIMIZATION RESULTS")
            print("=" * 50)
            print(f"Predicted minimum achievable error: {min_error:.3f}%")
            print(f"Current average error: {self.y.mean():.3f}%")
            print(f"Potential improvement: {self.y.mean() - min_error:.3f}%")
            print("\nRecommended parameter adjustments:")
            print("-" * 50)
            
            adjustments_made = False
            for feature in self.top_features:
                current_median = self.X[feature].median()
                optimal = optimal_settings[feature]
                change = optimal - current_median
                change_pct = (change / current_median) * 100
                
                # Only show adjustments that are meaningful (>0.1% change)
                if abs(change_pct) > 0.1:
                    adjustments_made = True
                    direction = "↑" if change > 0 else "↓"
                    print(f"{feature:20s}: {current_median:8.3f} → {optimal:8.3f} ({change_pct:+.1f}%) {direction}")
            
            if not adjustments_made:
                print("No meaningful adjustments found. The model may need more data or different features.")
                print("Current vs optimal values (showing all):")
                for feature in self.top_features[:5]:
                    current_median = self.X[feature].median()
                    optimal = optimal_settings[feature]
                    change_pct = ((optimal - current_median) / current_median) * 100
                    print(f"{feature:20s}: {current_median:8.3f} → {optimal:8.3f} ({change_pct:+.1f}%)")
            
            return optimal_settings, min_error
        else:
            print("Optimization failed. Using best found solution.")
            return dict(zip(self.top_features, best_result.x)), best_result.fun

    def line_specific_analysis_utilities(self, line_id, top_k=10):
        """Utility: Analyze specific production line"""
        if self.X is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if line_id >= len(self.X):
            print(f"Error: Line ID {line_id} not found. Available: 0-{len(self.X)-1}")
            return None
            
        # Ensure optimization model is trained
        if self.optimization_model is None:
            self.optimization_utilities(top_k=top_k)
        
        current_settings = self.X.iloc[line_id][self.top_features]
        current_error = self.y.iloc[line_id]
        
        # Optimize for this specific line
        bounds = self.constraint_utilities()
        
        result = minimize(
            self.optimization_objective,
            current_settings.values,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500}
        )
        
        if result.success:
            optimal_settings = dict(zip(self.top_features, result.x))
            min_error = result.fun
            
            print(f"\n📊 ANALYSIS FOR PRODUCTION LINE {line_id}")
            print("=" * 50)
            print(f"Current error: {current_error:.3f}%")
            print(f"Achievable error: {min_error:.3f}%")
            print(f"Improvement potential: {current_error - min_error:.3f}%")
            
            print("\nKey parameter adjustments needed:")
            print("-" * 45)
            
            adjustments_made = False
            for feature in self.top_features[:5]:  # Show top 5 most important
                current = current_settings[feature]
                optimal = optimal_settings[feature]
                change = optimal - current
                change_pct = (change / current) * 100 if current != 0 else 0
                
                # Only show meaningful adjustments
                if abs(change_pct) > 0.1:
                    adjustments_made = True
                    direction = "↑" if change > 0 else "↓"
                    print(f"{feature:20s}: {current:8.3f} → {optimal:8.3f} ({change_pct:+.1f}%) {direction}")
            
            if not adjustments_made:
                print("No meaningful adjustments needed for top parameters.")
                print("Current values are already near optimal.")
                for feature in self.top_features[:3]:
                    current = current_settings[feature]
                    optimal = optimal_settings[feature]
                    change_pct = ((optimal - current) / current) * 100
                    print(f"{feature:20s}: {current:8.3f} → {optimal:8.3f} ({change_pct:+.1f}%)")
            
            return optimal_settings
        else:
            print(f"Optimization failed for line {line_id}")
            return None

    def diagnostic_analysis(self):
        """Run diagnostic analysis to understand why no adjustments are being made"""
        print("\n🔍 DIAGNOSTIC ANALYSIS")
        print("=" * 50)
        
        if self.optimization_model is None:
            print("No model trained yet. Run optimization first.")
            return
            
        # Check model sensitivity
        medians = self.X[self.top_features].median().values
        base_prediction = self.optimization_model.predict([medians])[0]
        
        print(f"Base prediction with median values: {base_prediction:.4f}")
        print("\nTesting parameter sensitivity (±10% changes):")
        
        for i, feature in enumerate(self.top_features[:5]):
            test_params_high = medians.copy()
            test_params_low = medians.copy()
            
            # Test +10% change
            test_params_high[i] *= 1.1
            pred_high = self.optimization_model.predict([test_params_high])[0]
            
            # Test -10% change
            test_params_low[i] *= 0.9
            pred_low = self.optimization_model.predict([test_params_low])[0]
            
            change_high = pred_high - base_prediction
            change_low = pred_low - base_prediction
            
            sensitivity = max(abs(change_high), abs(change_low))
            sensitivity_indicator = "HIGH" if sensitivity > 0.1 else "LOW"
            
            print(f"{feature:20s}: +10% → {change_high:+.4f}, -10% → {change_low:+.4f} [{sensitivity_indicator}]")

def create_realistic_production_data():
    """Create more realistic production data with clear relationships"""
    np.random.seed(42)
    n_lines = 40
    n_params = 2500
    
    # Create base parameters with different distributions
    X = pd.DataFrame()
    
    for i in range(n_params):
        # Mix of different parameter distributions
        if i % 5 == 0:
            # Some parameters with normal distribution
            X[f'param_{i:04d}'] = np.random.normal(100, 15, n_lines)
        elif i % 5 == 1:
            # Some with uniform distribution
            X[f'param_{i:04d}'] = np.random.uniform(50, 150, n_lines)
        else:
            # Most with normal but different parameters
            X[f'param_{i:04d}'] = np.random.normal(80 + (i % 20) * 2, 10 + (i % 5), n_lines)
    
    # Create meaningful non-linear relationships with the target
    # Select 20 parameters to actually affect the error
    important_params = [
        'param_0042', 'param_0127', 'param_0256', 'param_0333', 'param_0489',
        'param_0567', 'param_0689', 'param_0721', 'param_0855', 'param_0999',
        'param_1123', 'param_1288', 'param_1422', 'param_1555', 'param_1678',
        'param_1822', 'param_1955', 'param_2088', 'param_2222', 'param_2388'
    ]
    
    # Create complex target relationships
    y = np.ones(n_lines) * 5.0  # Base error
    
    for param in important_params[:10]:
        # Quadratic relationships for some parameters
        y += 0.01 * (X[param] - X[param].mean()) ** 2 * np.random.uniform(-1, 1)
    
    for param in important_params[10:15]:
        # Linear relationships with interaction effects
        y += 0.05 * (X[param] - X[param].mean()) * np.random.normal(0, 0.5)
    
    for param in important_params[15:]:
        # Optimal range relationships (U-shaped or inverted U-shaped)
        optimal_value = X[param].mean()
        y += 0.02 * np.abs(X[param] - optimal_value) * np.random.uniform(0.5, 2.0)
    
    # Add noise
    y += np.random.normal(0, 1.0, n_lines)
    
    # Ensure positive error values
    y = np.maximum(y, 0.5)
    
    print(f"Created realistic data with {len(important_params)} truly important parameters")
    return X, y, important_params

def main():
    """Main function to run the production line optimizer"""
    print("🏭 PRODUCTION LINE OPTIMIZATION SYSTEM")
    print("=" * 50)
    
    try:
        # Create more realistic data
        print("Generating realistic production data...")
        X, y, true_important = create_realistic_production_data()
        
        # Initialize optimizer
        optimizer = ProductionOptimizer()
        optimizer.load_data(X, y)
        
        # Step 1: Feature importance analysis
        print("\n1. 🔍 ANALYZING FEATURE IMPORTANCE...")
        importance = optimizer.analyze_feature_importance()
        print("Top 10 most influential parameters:")
        for i, (param, row) in enumerate(importance.head(10).iterrows(), 1):
            is_true_important = "✓" if param in true_important else " "
            print(f"   {i:2d}. {is_true_important} {param} (score: {row['combined_score']:.4f})")
        
        # Step 2: Global optimization
        print("\n2. ⚙️  FINDING OPTIMAL PARAMETER SETTINGS...")
        optimal_settings, min_error = optimizer.find_optimal_parameters(top_k=10)
        
        # Step 3: Diagnostic analysis
        optimizer.diagnostic_analysis()
        
        # Step 4: Line-specific analysis
        print("\n3. 🔧 LINE-SPECIFIC OPTIMIZATION...")
        for line_id in [0, 5, 10]:  # Analyze a few sample lines
            optimizer.line_specific_analysis_utilities(line_id)
            
        # Step 5: Visualization
        print("\n4. 📈 GENERATING VISUALIZATIONS...")
        optimizer.visualization_utilities()
            
        print("\n✅ Optimization completed!")
        
        # Add this line to keep the program running
        input("\nPress Enter to exit...")  # This keeps the program alive until you press Enter
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
