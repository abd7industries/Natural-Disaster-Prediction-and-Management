import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ----------- PART 1: Sensor Simulation and Risk Analysis -----------

def simulate_sensor_data():
    return {
        'Temperature (°C)': round(random.uniform(20, 45), 1),
        'Humidity (%)': round(random.uniform(30, 100), 1),
        'Rainfall (mm)': round(random.uniform(0, 500), 1),
        'Wind Speed (km/h)': round(random.uniform(0, 40), 1),
        'Seismic Activity': round(random.uniform(0.0, 6.0), 1)
    }

def calculate_risk(data):
    risk = {}
    risk['Temperature'] = min(max((data['Temperature (°C)'] - 35) * 3, 0), 100)
    risk['Humidity'] = min(max((100 - data['Humidity (%)']) * 1.1, 0), 100)
    risk['Rainfall'] = min((data['Rainfall (mm)'] / 5), 100)
    risk['Wind Speed'] = min(data['Wind Speed (km/h)'] * 2.5, 100)
    risk['Seismic Activity'] = min(data['Seismic Activity'] * 20, 100)
    return risk

def predict_disaster(risk):
    if risk['Seismic Activity'] > 80:
        return "Earthquake Alert"
    elif risk['Rainfall'] > 70 and risk['Humidity'] > 60:
        return "Flood Alert"
    elif risk['Wind Speed'] > 70 and risk['Temperature'] > 50:
        return "Cyclone Alert"
    elif risk['Temperature'] > 60 and risk['Humidity'] > 50:
        return "Wildfire Risk"
    else:
        return "No Disaster Detected"

def plot_risk_chart(risk_data):
    parameters = list(risk_data.keys())
    risk_values = list(risk_data.values())
    plt.figure(figsize=(12, 6))
    bar = plt.bar(parameters, risk_values, color='tomato', edgecolor='black')
    plt.ylim(0, 100)
    plt.title("Environmental Disaster Risk Levels", fontsize=16)
    plt.xlabel("Indicators")
    plt.ylabel("Risk Level (%)")
    for i in bar:
        plt.text(i.get_x() + bar[0].get_width()/2.0, i.get_height() + 2,
                 f"{i.get_height():.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

# ----------- PART 2: Machine Learning Classification -----------

def disaster_classifier():
    np.random.seed(42)
    data_size = 500
    data = pd.DataFrame({
        'temperature': np.random.normal(30, 5, data_size),
        'humidity': np.random.normal(70, 10, data_size),
        'seismic_activity': np.random.uniform(0, 10, data_size),
        'rainfall_level': np.random.uniform(0, 200, data_size),
        'disaster': np.random.choice([0, 1], data_size, p=[0.8, 0.2])
    })

    X = data[['temperature', 'humidity', 'seismic_activity', 'rainfall_level']]
    y = data['disaster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Machine Learning Classification ---")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Feature importance plot
    plt.figure(figsize=(8, 5))
    feature_importance = model.feature_importances_
    plt.barh(X.columns, feature_importance, color='green')
    plt.title('Feature Importance in Disaster Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# ----------- PART 3: Regional Disaster Risk Visualization -----------

def visualize_region_risks():
    np.random.seed(42)
    data = {
        'Region': ['North', 'South', 'East', 'West', 'Central'],
        'Earthquake Risk': np.random.randint(20, 90, 5),
        'Flood Risk': np.random.randint(10, 70, 5),
        'Wildfire Risk': np.random.randint(5, 60, 5),
        'Hurricane Risk': np.random.randint(15, 80, 5),
        'Tsunami Risk': np.random.randint(10, 50, 5)
    }

    df = pd.DataFrame(data)
    df.set_index('Region', inplace=True)

    # Stacked bar chart
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Stacked Bar Chart: Natural Disaster Risks by Region')
    plt.xlabel('Region')
    plt.ylabel('Risk Level (%)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Pie chart for average distribution
    avg_risks = df.mean()
    plt.figure(figsize=(8, 8))
    plt.pie(avg_risks, labels=avg_risks.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('pastel'))
    plt.title('Pie Chart: Average Natural Disaster Risks Distribution')
    plt.tight_layout()
    plt.show()

    # Heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Heatmap: Natural Disaster Risk Levels by Region')
    plt.tight_layout()
    plt.show()

# ----------- Main Execution -----------

def main():
    print("Simulating sensor data...\n")
    sensor_data = simulate_sensor_data()
    for key, val in sensor_data.items():
        print(f"{key}: {val}")

    risk_levels = calculate_risk(sensor_data)
    print("\nCalculated Risk Levels:")
    for key, val in risk_levels.items():
        print(f"{key}: {val:.1f}")

    prediction = predict_disaster(risk_levels)
    print("\nPrediction:", prediction)

    plot_risk_chart(risk_levels)
    disaster_classifier()
    visualize_region_risks()

if __name__ == "__main__":
    main()
