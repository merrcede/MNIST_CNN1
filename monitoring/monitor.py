import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab, NumTargetDriftTab

def generate_dashboard(reference_data, current_data, column_mapping):
    # Build a dashboard using Evidently
    dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab(), NumTargetDriftTab()])
    dashboard.calculate(reference_data, current_data, column_mapping)
    dashboard.save('model_monitoring_report.html')  # ذخیره داشبورد به صورت یک فایل HTML

column_mapping = {
    'target': 'target_column_name',
    'prediction': 'prediction_column_name',
    'numerical_features': ['feature1', 'feature2', ...],
    'categorical_features': ['feature3', 'feature4', ...],
}
current_accuracy = 0.85  # This value should be updated based on recent model evaluations

# Record precision and set warning
def check_model_performance(current_accuracy, threshold=0.90):
    if current_accuracy < threshold:
        print(f"Warning: The accuracy of the model has reached below {threshold}!")


check_model_performance(current_accuracy)
