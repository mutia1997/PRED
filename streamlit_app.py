import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
@st.cache
def load_data(kecamatan):
    filename = kecamatan.lower().replace(" ", "_") + ".csv"
    df = pd.read_csv(filename)
    return df

# Split data and train model
def train_model(df, test_size, random_state):
    X = df[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]
    y = df['harga']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Grid search for hyperparameters
def grid_search_model(df, test_size, random_state):
    X = df[['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_bangunan']]
    y = df['harga']
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30]
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, X_test, y_test

def main():
    st.title('Prediksi Harga Apartemen di Jakarta Pusat')
    kecamatan = st.sidebar.selectbox('Pilih Kecamatan', ['Tanah Abang', 'Menteng', 'Cempaka Putih', 'Senen', 'Kemayoran', 'Gambir'])
    test_size = 0.05
    random_state = 42
    
    if kecamatan == 'Menteng':
        test_size = 0.25
        random_state = 60
        model, X_test, y_test = grid_search_model(load_data(kecamatan), test_size, random_state)
    elif kecamatan == 'Kemayoran':
        test_size = 0.05
        random_state = 42
        model, X_test, y_test = train_model(load_data(kecamatan), test_size, random_state)
    elif kecamatan == 'Cempaka Putih':
        test_size = 0.1
        random_state = 42
        model, X_test, y_test = grid_search_model(load_data(kecamatan), test_size, random_state)
    elif kecamatan == 'Senen':
        test_size = 0.2
        random_state = 42
        model, X_test, y_test = grid_search_model(load_data(kecamatan), test_size, random_state)
    elif kecamatan == 'Gambir':
        test_size = 0.2
        random_state = 42
        model, X_test, y_test = grid_search_model(load_data(kecamatan), test_size, random_state)
    else:
        test_size = 0.05
        random_state = 42
        model, X_test, y_test = grid_search_model(load_data(kecamatan), test_size, random_state)

    st.write("Model terbaik:")
    st.write(model)

    st.write("MSE (Mean Squared Error) pada data uji:")
    st.write(mean_squared_error(y_test, model.predict(X_test)))

if __name__ == '__main__':
    main()