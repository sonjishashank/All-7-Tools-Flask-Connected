from flask import Flask, request, jsonify
from flask_cors import CORS
import dask.dataframe as dd
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_plots(selected_district):
    # Read the CSV file using Dask
    df = dd.read_csv('Tool2.csv')

    # Filter the DataFrame based on the selected district
    df_selected_district = df[df['District_Name'] == selected_district]

    # Check if the selected district has any data
    if len(df_selected_district.index) == 0:
        return {"error": "No data available for the selected district"}

    # Group the filtered DataFrame by 'UnitName' and 'beat' and calculate the size of each group
    df_crime = df_selected_district.groupby(['UnitName', 'beat']).size().reset_index().rename(columns={0: 'crime_count'}).compute()

    # List of unit names to plot
    unit_names_to_plot = df_crime['UnitName'].unique()

    # Create subplots for each unit name
    num_rows = len(unit_names_to_plot)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, num_rows*5))

    # Iterate over each unit name and plot beat-wise crime distribution
    for i, unit_name in enumerate(unit_names_to_plot):
        # Filter data for the current unit name
        data_unit = df_crime[df_crime['UnitName'] == unit_name]
        
        # Extract beat and crime count data
        x = data_unit['beat']
        y = data_unit['crime_count']

        # Plot beat-wise crime distribution for the current unit name
        axes[i].bar(x, y)
        axes[i].set_title(unit_name)  # Set subplot title as unit name
        axes[i].set_xlabel('Beat')
        axes[i].set_ylabel('Number of Crimes')
        axes[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability
        axes[i].grid(which='both', linestyle=':')

    # Adjust layout
    plt.tight_layout()

    # Save the plots as a PDF file with the district name
    output_file = f"{selected_district.lower()}_crime_distribution_plots.pdf"
    plt.savefig(output_file)

    # Close the plot to prevent it from displaying in the console
    plt.close()

    # Convert plot data to JSON format
    plots_json = {"Tool2.csv": output_file}

    return plots_json

@app.route('/crime_distribution/<district>', methods=['GET'])
def crime_distribution(district):
    # List of valid districts
    valid_districts = ['Bagalkot', 'Ballari', 'Belagavi City', 'Belagavi Dist',
                       'Bengaluru City', 'Bengaluru Dist', 'Bidar', 'Chamarajanagar',
                       'Chickballapura', 'Chikkamagaluru', 'Chitradurga', 'CID',
                       'Coastal Security Police', 'Dakshina Kannada', 'Davanagere',
                       'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Hubballi Dharwad City',
                       'K.G.F', 'Kalaburagi', 'Kalaburagi City', 'Karnataka Railways',
                       'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mangaluru City',
                       'Mysuru City', 'Mysuru Dist', 'Raichur', 'Ramanagara',
                       'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada',
                       'Vijayanagara', 'Vijayapur', 'Yadgir']

    if district in valid_districts:
        # Generate plots
        plots_data = generate_plots(district)

        # Return JSON response
        return jsonify(plots_data)
    else:
        return jsonify({"error": "Invalid district name"}), 400

if __name__ == '__main__':
    app.run(debug=True)
