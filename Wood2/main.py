import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import joblib
import tensorflow as tf
import datetime as dt
import requests
from twilio.rest import Client

base_url="http://api.openweathermap.org/data/2.5/weather?"
api_key="7b638614d930ef7dcf19ba374ad61dc0"
city="Pune"#You can change the city
url=base_url+"appid="+api_key+"&q="+city
response=requests.get(url).json()

# Load dataset
file_path = 'dataset_h12345.csv'
ds = pd.read_csv(file_path)

# Example target mapping with existing categories
category_mapping = {1:0, 2: 1, 3: 2, 4: 3, 5: 4}
ds['MAX CATEGORY'] = ds['MAX CATEGORY'].map(category_mapping)

# Check for any unmapped categories
if ds['MAX CATEGORY'].isnull().any():
    raise ValueError("Some categories in 'MAX CATEGORY' could not be mapped. Please check the category_mapping.")

# Split data into features and target
x = ds.drop(columns=['MAX CATEGORY'])
y = ds['MAX CATEGORY']

# Verify the number of unique categories
unique_categories = len(category_mapping)
assert len(np.unique(y)) == unique_categories, "Mismatch in number of unique categories after mapping."

# Convert target to one-hot encoding
num_classes = unique_categories  # Get number of unique classes
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Save the scaler
joblib.dump(scaler, 'scaler10.pkl')

# Calculate class weights
y_train_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(class_weights))

# Define the model for multi-class classification
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model with categorical_crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), class_weight=class_weights)

# Evaluate the model
test_loss, test_accu = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accu}")

# Save the model
model.save('calamity_prediction_model10sd.h5')

# Function to process input data and make a prediction
def predict_category(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure input data has the correct columns and order
    input_columns = x.columns  # Get the feature columns from the training data
    input_df = input_df.reindex(columns=input_columns, fill_value=0)  # Fill missing columns with 0

    # Check the shape of the input data
    if input_df.shape[1] != x_train.shape[1]:
        raise ValueError(f"Input data shape {input_df.shape[1]} does not match model input shape {x_train.shape[1]}")

    # Scale the input data using the trained scaler
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_scaled)

    # Get the predicted category (the index of the max probability)
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Map the predicted index back to the original category
    inverse_category_mapping = {v: k for k, v in category_mapping.items()}
    predicted_category = inverse_category_mapping.get(predicted_index, 'Unknown')

    return predicted_category

# Example input data (ensure the order of features matches the model training order)
input_data = {
    'MAX WIND SPEED': (int(response['wind']['speed'])*1.94384),
    'MIN PRESSURE': response['main']['pressure'],
    # Add all other features used in the model, if any
}

# Get the prediction
predicted_category = predict_category(input_data)
print(f"Predicted Category: {predicted_category}")

# Twilio configuration
account_sid = 'AC9099abdae31954c928738815d55d0008'
auth_token = 'a8138a6a27b115707a15fec26813fd69'
client = Client(account_sid, auth_token)

# Define actions based on predicted category
if predicted_category == (5):
    print("Code: Red - Hurricane")
    print(response['weather'][0]['main'])
    print(response['weather'][0]['description'])
    
elif predicted_category == 4:
    message = "Code: Orange - Cyclone "
    print(response['weather'][0]['main'])
    print(response['weather'][0]['description'])

elif predicted_category == 3:
    message = "Code: Yellow - Heavy Winds"
    print(response['weather'][0]['main'])
    print(response['weather'][0]['description'])

elif predicted_category == 2:
    message = "Code: Green - Tropical Storm"
    print(response['weather'][0]['main'])
    print(response['weather'][0]['description'])

elif predicted_category == 1:
    message = "Code: White - Normal"
    print(response['weather'][0]['main'])
    print(response['weather'][0]['description'])

 #Send SMS if necessary
if predicted_category in [1, 2, 3, 4, 5]:  # Only send SMS for categories 3 and above
    sms = f"SOS: {message}"
    twilio_message = client.messages.create(
        from_='+19382018409',
        body=sms,
        to='+919930400475'#Add your own phone number
    )
    print(f"Message sent: {twilio_message.sid}")
else:
    print("No SMS sent. Situation is not critical.")