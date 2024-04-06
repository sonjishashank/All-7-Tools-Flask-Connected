import base64
from io import BytesIO
import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)

# Load the dataset with Dask
ddf = dd.read_csv('Tool5.csv')

# Convert Dask DataFrame to Pandas DataFrame for plotting
df = ddf.compute()

# Create a static6 folder if it doesn't exist
static_folder = 'static6'
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

@app.route('/predict/<district>', methods=['GET'])
def predict(district):
    # Filter the dataset for the specified district
    district_data = df[df['District_Name'].str.lower() == district.lower()]
    
    if district_data.empty:
        return jsonify({"error": f"No data found for district {district}"}), 404
    
    # Plot the trend of crime occurrence by age group
    plt.figure(figsize=(10, 6))
    sns.countplot(x='age', data=district_data, palette='muted')
    plt.title('Crime Occurrence by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Victims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    age_plot_file = save_plot_to_file('age_plot')
    plt.close()

    # Plot the trend of crime occurrence by sex
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', data=district_data, palette='muted')
    plt.title('Crime Occurrence by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Number of Victims')
    plt.tight_layout()
    sex_plot_file = save_plot_to_file('sex_plot')
    plt.close()

    # Plot the trend of crime occurrence by location (District_Name)
    plt.figure(figsize=(14, 8))
    sns.countplot(x='District_Name', data=district_data, palette='muted')
    plt.title('Crime Occurrence by Location (District)')
    plt.xlabel('District')
    plt.ylabel('Number of Victims')
    plt.xticks(rotation=90)
    plt.tight_layout()
    district_plot_file = save_plot_to_file('district_plot')
    plt.close()

    return jsonify({
        "age_plot": f"/static6/{age_plot_file}",
        "sex_plot": f"/static6/{sex_plot_file}",
        "district_plot": f"/static6/{district_plot_file}"
    })

def save_plot_to_file(plot_name):
    """Save the current plot to a PNG file."""
    file_path = os.path.join(static_folder, f"{plot_name}.png")
    plt.savefig(file_path, format='png')
    return f"{plot_name}.png"

if __name__ == '__main__':
    app.run(debug=True)
