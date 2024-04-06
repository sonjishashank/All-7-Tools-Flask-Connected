import base64
from io import BytesIO
from flask import Flask, jsonify
from flask_cors import CORS
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)

# Load the dataset with Dask, specifying the dtype for the 'age' column
ddf = dd.read_csv('Tool4.csv', dtype={'age': 'object'})

# Convert Dask DataFrame to Pandas DataFrame for plotting
df = ddf.compute()

@app.route('/predict', methods=['GET'])
def predict():
    # Adjust age probability values to increase importance
    age_probabilities = {
        '0-20': 0.1,
        '21-30': 0.3,
        '31-40': 0.5,
        '41-50': 0.6,
        '51-60': 0.5,
        '61-70': 0.3,
        '71+': 0.1
    }

    # Calculate profession occurrence counts
    profession_counts = df['Profession'].value_counts()

    # Select the top 15 occurring professions
    top_professions = profession_counts.head(15).index.tolist()

    # Assign values to professions based on occurrence counts
    profession_values = {}
    max_value = 20
    min_value = 1
    max_count = profession_counts.max()
    min_count = profession_counts.min()
    for profession, count in profession_counts.items():
        if profession in top_professions:
            scaled_value = min_value + (count - min_count) * (max_value - min_value) / (max_count - min_count)
            profession_values[profession] = scaled_value

    # Clean up 'AgeGroup' column and ensure all values are strings
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Considering gender distribution
    total_accused = len(df)
    male_count = df['Sex'].value_counts().get('MALE', 0)
    female_count = df['Sex'].value_counts().get('FEMALE', 0)
    male_probability = male_count / total_accused
    female_probability = female_count / total_accused

    # Define the additional contribution for gender
    male_contribution = 0.1  # Adjust as needed
    female_contribution = 0.05  # Adjust as needed

    # Group age into predefined age ranges
    age_bins = [0, 20, 30, 40, 50, 60, 70, np.inf]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

    # Combine probabilities
    df['CriminalityProbability'] = (df['AgeGroup'].map(age_probabilities) + 
                                    df['Profession'].map(profession_values) +
                                    np.where(df['Sex'] == 'MALE', male_contribution, female_contribution))

    # Generate predictions
    threshold_probability = 15  # Adjust threshold as needed
    df['Prediction'] = df['CriminalityProbability'] >= threshold_probability

    # Output results in terms of labels based on threshold probability
    def label_probability(probability):
        if probability >= threshold_probability:
            return "Most likely"
        else:
            return "Not likely"

    # Apply the function to the CriminalityProbability column
    df['Prediction'] = df['CriminalityProbability'].apply(label_probability)

    # View criminal probability by district
    plt.figure(figsize=(12, 6))
    sns.barplot(x='District_Name', y='CriminalityProbability', data=df, estimator=np.mean)
    plt.title('Criminal Probability by District')
    plt.xlabel('District')
    plt.ylabel('Criminal Probability')
    plt.xticks(rotation=90)
    district_plot_file = save_plot_to_base64(plt)
    plt.close()

    return jsonify({"district_plot": district_plot_file})

def save_plot_to_base64(plot):
    """Save the current plot to a base64 encoded string."""
    buffer = BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    plot_bytes = buffer.getvalue()
    plot_base64 = base64.b64encode(plot_bytes).decode('utf-8')
    return plot_base64

if __name__ == '__main__':
    app.run(debug=True)
