import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

pd.options.mode.chained_assignment = None  # default='warn'

stocks_df = pd.read_csv("SQQQ.csv",
                        dtype={"Close":float, "Volume":int}, parse_dates=["Date"])

# Get your training data
training_data = stocks_df.iloc[:, 1:3].values


# Scale the data
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_data)

# Create the target data by comparing current day price to next day price
X = []
y = []

for i in range(1, len(stocks_df)):
    X.append(training_set_scaled[i-1:i, 0])
    y.append(training_set_scaled[i, 0])

# Convert to array format
X = np.asarray(X)
y = np.asarray(y)

# Split the data into training set and testing set
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshape the data from 1D to 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Input dimension (1,1)
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

# Define LSTM netwrok
x = keras.layers.LSTM(150, return_sequences=True)(inputs)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
# x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs = inputs, outputs = outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train, 
    epochs = 20, 
    batch_size = 32, 
    validation_split=0.2
)

predicted = model.predict(X)

test_predicted = []

for i in predicted:
    test_predicted.append(i[0])


df_predicted = stocks_df[1:][['Date']]

df_predicted["Predictions"] = test_predicted

close = []

for i in training_set_scaled:
    close.append(i[0])

df_predicted["Close"] = close[1:]


def interactive_plot(data, title):
    fig = px.line(title=title)
    fig.add_scatter(x = data["Date"], y=data["Predictions"], name = "Predictions")
    fig.add_scatter(x = data["Date"], y=data["Close"], name = "Actual")
    fig.show()

interactive_plot(df_predicted, "Actual vs Prediction")
