from flask import Flask, request, jsonify
from flask_cors import CORS
import dask.dataframe as dd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Assuming 'Tool6.csv' is structured correctly and available in the same directory
ddf = dd.read_csv('Tool6.csv')

# Compute necessary transformations upfront to improve runtime performance
# Make sure the dataset is not too large to fit into memory after computation
df_analyse = ddf.groupby(['Year', 'District_Name', 'Beat_Name']).size().compute().reset_index()
df_analyse.columns = ['Year', 'District_Name', 'Beat_Name', 'Number_of_crimes']

# Ensure uniform case and whitespace handling
df_analyse['District_Name'] = df_analyse['District_Name'].str.strip().str.lower()

# Pre-compute unique years and districts for quick lookup
unique_years = df_analyse['Year'].unique()
unique_districts = df_analyse['District_Name'].unique()

@app.route('/crime_ranking/<year>/<district>', methods=['GET'])
def crime_ranking(year, district):
    try:
        # Convert year to integer for matching
        year = int(year.strip())
    except ValueError:
        return jsonify({"error": "Invalid year format. Please use a valid year."}), 400

    # Normalize district input
    district = district.strip().lower()

    # Check if the year and district exist in the dataset
    if year not in unique_years or district not in unique_districts:
        return jsonify({"error": "Data for the preferred year or district not found."}), 404

    # Filter data for the selected year and district
    filtered_data = df_analyse[(df_analyse['Year'] == year) & (df_analyse['District_Name'] == district)]

    if filtered_data.empty:
        return jsonify({"error": "No data found for the selected year and district."}), 404

    # Sort and rank the beats based on the number of crimes
    ranked_beats = filtered_data.sort_values(by='Number_of_crimes', ascending=False)

    ranking_list = [{
        "rank": idx + 1,
        "beat": row['Beat_Name'],
        "number_of_crimes": row['Number_of_crimes']
    } for idx, row in ranked_beats.iterrows()]

    response = {
        "year": year,
        "district": district,
        "ranking": ranking_list
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)