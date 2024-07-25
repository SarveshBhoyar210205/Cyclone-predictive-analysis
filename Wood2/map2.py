import requests
import folium
import webbrowser
import os
import tempfile

# Fetch coordinates from the Flask server
response = requests.get('http://127.0.0.1:5000/get_location')
data = response.json()
latitude = data['latitude']
longitude = data['longitude']

# Check if the coordinates are available
if latitude is not None and longitude is not None:
    # Create a map centered at the fetched location
    map_center = [latitude, longitude]
    my_map = folium.Map(location=map_center, zoom_start=13)

    print(latitude)
    print(longitude)
    # Add a marker
    folium.Marker([latitude, longitude], popup='Your Location').add_to(my_map)

    folium.Marker(
        location=[latitude, longitude],
        popup="Pune",
        icon=folium.Icon(icon="cloud"),
    ).add_to(my_map)

    folium.Circle(
    radius=300,
    location=[latitude, longitude],
    popup="Pune",
    color="blue",
    fill=True,
    fill_color="blue"
    ).add_to(my_map)  
    

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        temp_filename = temp_file.name
        my_map.save(temp_filename)

    # Open the map in the default web browser
    webbrowser.open('file://' + os.path.realpath(temp_filename))
   


    # Optionally, delete the temporary file after opening (uncomment if desired)
    # os.remove(temp_filename)
else:
    print("Coordinates not available. Please send location data first.")


