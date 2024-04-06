from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
import requests
import plotly.graph_objects as go
import dask.dataframe as dd

app = Flask(__name__)
CORS(app)

@app.route('/map')
def show_map():
    # Load your dataset using Dask
    data = dd.read_csv('Tool7.csv')

    # Define features (latitude and longitude)
    X = data[['Latitude', 'Longitude']]

    # Drop rows with missing values
    X = X.dropna()

    # Choose the number of clusters (future locations)
    num_clusters = 5

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the model on the latitude and longitude data
    kmeans.fit(X)

    # Predict the centroids of the clusters as future latitude and longitude
    future_locations = kmeans.cluster_centers_

    # Convert future_locations to a DataFrame
    future_locations_df = pd.DataFrame(future_locations, columns=['Latitude', 'Longitude'])

    # Function to get the nearest police station for a given latitude and longitude
    def get_nearest_police_station(latitude, longitude, api_key):
        url = f"https://dev.virtualearth.net/REST/v1/Locations/{latitude},{longitude}?key={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and "resourceSets" in data and len(data["resourceSets"]) > 0:
            resources = data["resourceSets"][0].get("resources", [])
            if resources:
                address = resources[0].get("name", "")
                coordinates = resources[0].get("point", {}).get("coordinates", [])
                if coordinates:
                    return address, coordinates[0], coordinates[1]
        
        print("Error fetching data from Bing Maps API")
        return None, None, None

    # Function to find the closest hospitals using Bing Maps API
    def find_closest_hospitals(latitude, longitude, api_key, radius=5000, max_results=1):
        url = f"https://dev.virtualearth.net/REST/v1/LocalSearch/?query=hospital&userLocation={latitude},{longitude}&radius={radius}&maxResults={max_results}&key={api_key}"
        response = requests.get(url)
        data = response.json()

        hospitals = []
        if "resourceSets" in data and len(data["resourceSets"]) > 0:
            for result in data["resourceSets"][0]["resources"]:
                name = result["name"]
                address = result["Address"]["formattedAddress"]
                lat = result["point"]["coordinates"][0]
                lng = result["point"]["coordinates"][1]
                hospitals.append((name, address, lat, lng))
        return hospitals

    # Your Bing Maps API key
    api_key = "AjMNQNjkQra1lSLBQp7QsXk-IqUfE9o-Ml1jPPfJuiQIlFx3EmM3fzAF5tXYyP_k"

    # Get the nearest police station and hospitals for each future location
    locations_info = []
    for index, row in future_locations_df.iterrows():
        # Predicted Crime Locations
        location_info = {
            'Type': 'Predicted Crime Location',
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude']
        }
        locations_info.append(location_info)

        # Police stations
        address, lat, lng = get_nearest_police_station(row['Latitude'], row['Longitude'], api_key)
        if lat is not None and lng is not None:
            location_info = {
                'Type': 'Police Station',
                'Name': address,
                'Latitude': lat,
                'Longitude': lng
            }
            locations_info.append(location_info)
        
        # Hospitals
        nearest_hospitals = find_closest_hospitals(row['Latitude'], row['Longitude'], api_key)
        for hospital in nearest_hospitals:
            location_info = {
                'Type': 'Hospital',
                'Name': hospital[0],
                'Address': hospital[1],
                'Latitude': hospital[2],
                'Longitude': hospital[3]
            }
            locations_info.append(location_info)

    # Create traces
    future_trace = go.Scattermapbox(
        lat=future_locations_df['Latitude'],
        lon=future_locations_df['Longitude'],
        mode='markers',
        marker=dict(size=10, color='red', opacity=0.7),
        name='Predicted Crime Locations'
    )

    police_trace = go.Scattermapbox(
        lat=[info['Latitude'] for info in locations_info if info['Type'] == 'Police Station'],
        lon=[info['Longitude'] for info in locations_info if info['Type'] == 'Police Station'],
        mode='markers',
        marker=dict(size=10, color='green', opacity=0.7),
        name='Police Stations'
    )

    hospital_trace = go.Scattermapbox(
        lat=[info['Latitude'] for info in locations_info if info['Type'] == 'Hospital'],
        lon=[info['Longitude'] for info in locations_info if info['Type'] == 'Hospital'],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.7),
        name='Hospitals'
    )

    # Map layout
    layout = go.Layout(
        title='Predicted Locations with Nearest Police Stations and Hospitals',
        mapbox=dict(
            style="open-street-map",
            zoom=10
        )
    )

    # Create figure and add traces
    fig = go.Figure(data=[future_trace, police_trace, hospital_trace], layout=layout)

    # Convert the figure to JSON
    map_json = fig.to_json()

    return jsonify(map_json)

if __name__ == '__main__':
    app.run(debug=True)
