import dask.dataframe as dd
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from flask import render_template, request

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the CSV data
crime_data = dd.read_csv('output_file.csv')  # Using Dask to read the CSV file
crime_data = crime_data.dropna(subset=['Latitude', 'Longitude'])
crime_data_pandas = crime_data.compute()

# Define the districts
districts = [
    'Bagalkot', 'Ballari', 'Belagavi City', 'Belagavi Dist',
    'Bengaluru City', 'Bengaluru Dist', 'Bidar', 'Chamarajanagar',
    'Chickballapura', 'Chikkamagaluru', 'Chitradurga', 'CID',
    'Coastal Security Police', 'Dakshina Kannada', 'Davanagere',
    'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Hubballi Dharwad City',
    'K.G.F', 'Kalaburagi', 'Kalaburagi City', 'Karnataka Railways',
    'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mangaluru City',
    'Mysuru City', 'Mysuru Dist', 'Raichur', 'Ramanagara',
    'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada',
    'Vijayanagara', 'Vijayapur', 'Yadgir'
]

from flask import request

@app.route('/api/maps/<district_name>', methods=['GET'])
def get_maps_by_district(district_name):
    if district_name not in districts:
        return jsonify({'error': 'District not found'}), 404

    district_data = crime_data_pandas[crime_data_pandas['District_Name'] == district_name]
    if district_data.empty:
        return jsonify({'error': 'No data available for the district'}), 404

    center = {'lat': district_data['Latitude'].mean(), 'lon': district_data['Longitude'].mean()}
    map_data = {
        'lat': district_data['Latitude'].tolist(),
        'lon': district_data['Longitude'].tolist(),
        'crimeType': district_data['CrimeType'].tolist()
    }
    return jsonify({'district': district_name, 'center': center, 'data': map_data})

if __name__ == '__main__':
    app.run(debug=True)
    