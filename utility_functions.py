def visualization_tools(optimizer):
    """Additional visualization tools"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Plot top feature importance
    plt.figure(figsize=(12, 8))
    top_10 = optimizer.feature_importance.head(10)
    
    plt.subplot(2, 2, 1)
    top_10['Overall_Score'].plot(kind='barh')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Overall Importance Score')
    
    plt.subplot(2, 2, 2)
    # Correlation heatmap for top features
    top_features = top_10.index.tolist()[:5]
    correlation_matrix = optimizer.X[top_features + ['error_percentage']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix - Top Features')
    
    plt.tight_layout()
    plt.show()

def export_recommendations(recommendations, filename='production_optimization_report.csv'):
    """Export optimization recommendations"""
    report_data = []
    
    for line_id, rec in recommendations.items():
        row = {
            'line_id': line_id,
            'current_error': rec['current_error'],
            'predicted_min_error': rec['predicted_min_error'],
            'improvement_potential': rec['improvement_potential']
        }
        
        # Add current and recommended settings
        for param in rec['current_settings'].keys():
            row[f'current_{param}'] = rec['current_settings'][param]
            row[f'recommended_{param}'] = rec['recommended_settings'][param]
            row[f'change_{param}'] = rec['recommended_settings'][param] - rec['current_settings'][param]
            
        report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(filename, index=False)
    print(f"Recommendations exported to {filename}")