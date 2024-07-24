from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store latitude and longitude
latitude = None
longitude = None

@app.route('/set_location', methods=['POST'])
def set_location():
    global latitude, longitude
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    print(f"Received Latitude: {latitude}, Longitude: {longitude}")
    return jsonify({'status': 'success', 'latitude': latitude, 'longitude': longitude})

@app.route('/get_location', methods=['GET'])
def get_location():
    global latitude, longitude
    return jsonify({'latitude': latitude, 'longitude': longitude})

@app.route('/status', methods=['GET'])
def status():
    if latitude is not None and longitude is not None:
        return jsonify({'status': 'data received', 'latitude': latitude, 'longitude': longitude})
    else:
        return jsonify({'status': 'no data received'})

if __name__ == '__main__':
    app.run(debug=True)