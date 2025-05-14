import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Simulated disaster prediction dataset
np.random.seed(42)
data = {
    'Region': ['North', 'South', 'East', 'West', 'Central'],
    'Earthquake Risk': np.random.randint(20, 90, 5),
    'Flood Risk': np.random.randint(10, 70, 5),
    'Wildfire Risk': np.random.randint(5, 60, 5),
    'Hurricane Risk': np.random.randint(15, 65, 5),
    'Tsunami Risk': np.random.randint(0, 50, 5)
}

df = pd.DataFrame(data)
df.set_index('Region', inplace=True)

# Generate a bar plot for disaster risks
plt.figure(figsize=(12, 6))
df.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Stacked Bar Chart: Natural Disaster Risks by Region')
plt.ylabel('Risk Level (%)')
plt.xlabel('Region')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Generate a pie chart for average disaster distribution
avg_risks = df.mean()
plt.figure(figsize=(8, 8))
plt.pie(avg_risks, labels=avg_risks.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette("pastel"))
plt.title('Pie Chart: Average Natural Disaster Risk Distribution')
plt.show()

# Generate a heatmap for risk levels across regions
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="YlGnBu", fmt="d")
plt.title('Heatmap: Natural Disaster Risk Levels by Region')
plt.show()
