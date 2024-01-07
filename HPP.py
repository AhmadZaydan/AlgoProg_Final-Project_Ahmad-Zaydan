import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
import streamlit as st

# Class for house price prediction
class HousePricePredictor:
    def __init__(self, model, imputer):
        self.model = model
        self.imputer = imputer

    # Method for preprocessing inputs
    def preprocess_input(self, user_input):
        input_df = pd.DataFrame([user_input])
        input_df['GRS'] = input_df['GRS'].map({'ADA': True, 'TIDAK ADA': False})
        input_df = pd.DataFrame(self.imputer.transform(input_df), columns=input_df.columns)
        return input_df

    # Method for predicting the price
    def predict_price(self, user_input):
        input_df = self.preprocess_input(user_input)

        input_df['GRS'] = input_df['GRS']
        input_df = pd.DataFrame(self.imputer.transform(input_df), columns=input_df.columns)
        
        # Make a prediction
        predicted_price = self.model.predict(input_df[['LT', 'LB', 'JKT','JKM','GRS']])[0]
        return predicted_price

# Load the dataset (Dataset from kaggle.com)
df = pd.read_csv('HARGA RUMAH JAKSEL(new).csv')

# Display the first few rows of the dataset
print(df.head())

# Erase the comas from the 'HARGA' (price) column and convert it into an integer
df['HARGA'] = df['HARGA'].str.replace(',', '').astype(np.int64)

# Print the head again to check the change
print(df.head())

# Convert the "HARGA" column to numeric
df['HARGA'] = pd.to_numeric(df['HARGA'], errors='coerce')

# Print the head again to check the change
print(df.head())

# Assuming 'GRS' is the boolean column 
# Change the "ADA" and "TIDAK ADA" to True and False respectedly
df['GRS'] = df['GRS'].map({'ADA': True, 'TIDAK ADA': False})

# Print the head again to check the change
print(df.head())

# 'HARGA' is the target variable and other columns are features
# LT = Land Area, LB = Building Area, JKT = Bedrooms, JKM = Bathrooms, GRS = Garage
# Adjust the column names accordingly based on your dataset
X = df[['LT', 'LB', 'JKT', 'JKM', 'GRS']]
y = df['HARGA']

# Handle missing values (Nan, Infinite, etc)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Create an instance of the HousePricePredictor class
house_price_predictor = HousePricePredictor(model, imputer)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2_score_result = r2_score(y_test, predictions)
print(f'R2 score: {r2_score_result}')

# Streamlit app
st.title('House Price Prediction App')
st.subheader('This app can predict price for a house in South Jakarta')

land_area = st.number_input('Enter area of the land: ', min_value=50, max_value=2000,value=100)
build_area = st.number_input('Enter area of the building: ', min_value=50, max_value=2000,value=100)
bedroom = st.number_input('Enter amount of bedroom: ', min_value=1,value=5)
bathroom = st.number_input('Enter amount of bathroom: ', min_value=1,value=5)
garage = st.radio('Is there a garage?:', ['Yes', 'No'])

def checker(garage):
    # Set a variable 
    grg = 'TIDAK ADA'

    if garage == 'Yes':
        grg = 'ADA'
    elif garage == 'No':
        grg = 'TIDAK ADA'
    return grg

# Function to seperate the zeros into 3 digits 
def moneyChange(price):
    mny = int(price)
    money = str(mny)
    n = len(money)
    start = 0
    finish = 3
    listMoney = []
    if n > 3:
        if n%3==0:
            for x in range(int(n/3)):
                listMoney.append(money[start:finish])
                start += 3
                finish += 3

        else:
            m = n%3
            listMoney.append(money[:m])
            copyMoney = money[m:]
            for x in range(int(len(copyMoney)/3)):
                listMoney.append(copyMoney[start:finish])
                start +=3
                finish +=3

        newMoney = ".".join(listMoney)

    else:
        newMoney=money

    return newMoney

# Output managements
if st.button('Generate Predicted Price'):
    user_input = {
        'LT': land_area,
        'LB': build_area,
        'JKT': bedroom,
        'JKM': bathroom,
        'GRS': checker(garage)
    }

    # State the predicted price
    predicted_price = house_price_predictor.predict_price(user_input)

    # Round the predicted price to a certain number of zeros
    rounded_price = round(predicted_price, -7)  # Adjust the number of zeros as needed

    # Display the predicted price
    st.subheader(f'Predicted Price: Rp{moneyChange(predicted_price)}')
    st.subheader(f'Rounded Price: Rp{moneyChange(rounded_price)}')


# Visualize predicted vs actual prices
plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'k--', lw=2, label="Identity Line")
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='red', label="Trend Line")
plt.savefig("HPP.jpg")
plt.show()

