from flask import Flask, send_file, jsonify
from flask_cors import CORS
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
CORS(app)

ddf = dd.read_csv('Tool3.csv', parse_dates=['date_time'], dayfirst=True)
# If the above doesn't work due to format issues, use the following:
# ddf['date_time'] = dd.to_datetime(ddf['date_time'], format='%d-%m-%Y %H:%M')

ddf['hour'] = ddf['date_time'].dt.hour
ddf['day_of_week'] = ddf['date_time'].dt.dayofweek
ddf['month'] = ddf['date_time'].dt.month
ddf['season'] = ddf['date_time'].dt.month % 12 // 3 + 1

if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/plot/<district_name>')
def plot_crime_occurrence(district_name):
    filtered_ddf = ddf[ddf['District_Name'].str.lower() == district_name.lower()]
    filtered_df = filtered_ddf.compute()

    if filtered_df.empty:
        return jsonify({"error": "No data found for the selected district."}), 404

    units = filtered_df['UnitName'].unique()
    time_periods = ['hour', 'day_of_week', 'month', 'season']
    plot_paths = []

    for unit in units:
        unit_data = filtered_df[filtered_df['UnitName'] == unit]
        for time_period in time_periods:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=time_period, data=unit_data, palette='viridis')
            plt.title(f'Crime Occurrence by {time_period.capitalize()} in {district_name.title()}, Unit: {unit}')
            plt.xlabel(time_period.capitalize())
            plt.ylabel('Number of Crimes')

            plot_filename = f"{district_name}_{unit}_{time_period}_plot.png"
            plot_path = os.path.join('static', plot_filename)
            plt.savefig(plot_path)
            plt.close()

            plot_paths.append(plot_filename)

    return jsonify({"images": plot_paths})

if __name__ == '__main__':
    app.run(debug=True)
