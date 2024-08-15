import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as msr
from sklearn.preprocessing import OneHotEncoder as oe


print("-------------------This is Hunar Intern Task 2 ------------------------")
print("-----------------------House Price Priduction--------------------------")
df = pd.read_csv(r'/content/house price data.csv')
df.isnull().sum()
#here we can see dataset is perfect or not null so......
df.info()
#feature on which price depand mainly 
ftr = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated','street','street','city','statezip']
x = df[ftr]
y =df['price']
#encode object type categroical data using onehotencoder in pandas 
x_encode = pd.get_dummies(x,columns=['street','city','statezip'],drop_first = True)
print(x_encode.columns)
#split dataset for train and test model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply linear regression model on the training dataset
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = model.predict(X_train)

# Check accuracy on the training set
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

# Predict on the test set
y_test_pred = model.predict(X_test)

# Check accuracy on the test set
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

# Display the results
print(f'Training RMSE: {train_rmse}')
print(f'Training R²: {train_r2}')
print(f'Test RMSE: {test_rmse}')
print(f'Test R²: {test_r2}')

# Function to test the model with a different dataset
def test_model_with_new_data(new_data_path):
    new_data = pd.read_csv(new_data_path)
    new_data_cleaned = new_data.dropna().drop_duplicates()
    X_new = new_data_cleaned[features]
    y_new = new_data_cleaned['price']
    
    y_new_pred = model.predict(X_new)
    
    new_rmse = mean_squared_error(y_new, y_new_pred, squared=False)
    new_r2 = r2_score(y_new, y_new_pred)
    
    print(f'New Data RMSE: {new_rmse}')
    print(f'New Data R²: {new_r2}')

# Example usage of testing with a new dataset
# new_data_path = 'path_to_new_data.csv'
# test_model_with_new_data(new_data_path)
