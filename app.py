import os

import joblib
from flask import Flask,request
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

class HealthNet(nn.Module):
    def __init__(self):
        super(HealthNet, self).__init__()
        self.fc1 = nn.Linear(in_features=6, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

@app.route('/')
def hello():
    return "Hello, Flask!"

@app.route('/api')
def get():
    # str(request.args['Male'])
    # url = 'https://i0.wp.com/theperfectcurry.com/wp-content/uploads/2021/09/PXL_20210830_005055409.PORTRAIT.jpg'
    url=request.args['url']
    csv_id = request.args['id']
    genderid = request.args['gid']
    weightid = request.args['wid']
    heightid = request.args['hid']
    exerciseid = request.args['eid']
    ageid = request.args['aid']
    # Fetch the image from the URL

    response = requests.get(url)
    save_folder = 'csv'  # Adjust the save folder path as needed
    save_name = 'calorie_'+csv_id+'.csv'
    if response.status_code == 200:
        # Create the save folder if it does not exist
        os.makedirs(save_folder, exist_ok=True)

        # Determine the image file name
        if save_name is None:
            save_name = os.path.basename(url)

        # Construct the full save path
        save_path = os.path.join(save_folder, save_name)

        # Write the image content to the file
        with open(save_path, 'wb') as file:
            file.write(response.content)

    # Load the dataset
    file_path = 'csv/calorie_'+csv_id+'.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path)

    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Ensure the data is sorted by date
    data = data.sort_values('date')

    # Extract the calorie values
    calories = data['calorie'].values
    dates = data['date'].values

    # Normalize the data
    calories_mean = np.mean(calories)
    calories_std = np.std(calories)
    calories_normalized = (calories - calories_mean) / calories_std

    # Prepare the data
    def create_dataset(data, time_step=1, prediction_horizon=2):
        X, Y = [], []
        for i in range(len(data) - time_step - prediction_horizon + 1):
            a = data[i:(i + time_step)]
            X.append(a)
            Y.append(data[i + time_step + prediction_horizon - 1])
        return np.array(X), np.array(Y)

    time_step = 10  # Number of previous time steps to consider
    prediction_horizon = 2  # Number of steps ahead to predict
    X, Y = create_dataset(calories_normalized, time_step, prediction_horizon)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    # Initialize model, loss function, and optimizer
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Predict future values
    model.eval()
    with torch.no_grad():
        future_steps = 5  # Number of days you want to predict into the future
        predictions = []
        input_seq = torch.tensor(calories_normalized[-time_step:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for _ in range(future_steps):
            pred = model(input_seq)
            predictions.append(pred.item())
            input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(-1)), dim=1)

    predictions = np.array(predictions).flatten() * calories_std + calories_mean
    finalcalorie=round((sum(predictions) / len(predictions)),2)
    model = HealthNet()
    model.load_state_dict(torch.load('health_model.pth'))
    model.eval()

    # Load the fitted scaler
    scaler = joblib.load('scaler.pkl')  # Ensure you have this filefinalcalorie

    # Example single input
    single_input = np.array([genderid,ageid,weightid,heightid,exerciseid,finalcalorie]).reshape(1, -1)  # Convert to 2D array for scaler

    # Scale the new data using the same scaler
    new_data_scaled = scaler.transform(single_input)

    # Convert to PyTorch tensor
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # Predict using the trained model
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(new_data_tensor)

    print("Predictions for new data:", predictions.item())

    return str(round(predictions.item(),2))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('love')
    app.run(port=5001)






