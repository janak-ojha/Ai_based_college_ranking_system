import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Function to preprocess and predict
def preprocess_and_predict(df, scaler, model):
    # Extract features for prediction (excluding 'college_name')
    features = df.drop(columns=['college_name'])

    # Transform features using the fitted scalery
    X_scaled = scaler.transform(features)

    # Predict scores using the trained model
    predicted_scores = model.predict(X_scaled)

    # Create a DataFrame to hold predicted scores
    predictions_df = pd.DataFrame({'college_name': df['college_name'], 'predicted_score': predicted_scores})

    # Sort colleges by predicted score in descending order
    ranked_colleges = predictions_df.sort_values(by='predicted_score', ascending=False)

    # Add a ranking column
    ranked_colleges['rank'] = range(1, len(ranked_colleges) + 1)

    return ranked_colleges[['college_name', 'rank']]

def load_data(file_path):
    # Load data from CSV file
    df = pd.read_csv(file_path)
    return df

def train_model(df):
    # Check if 'true_scores' column exists
    if 'true_scores' in df.columns:
        # Split the data into training and testing sets
        X = df.drop(columns=['college_name', 'true_scores'])
        y = df['true_scores']
        
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X)
        
        # Initialize RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y)

        return scaler, model
    else:
        st.error("Column 'true_scores' not found in the uploaded CSV file.")

def main():
    st.title('College Ranking Prediction App')

    st.markdown('Upload a CSV file containing data for colleges to predict their rankings.')

    # File upload for testing
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded CSV file
        df_test = load_data(uploaded_file)

        # Display the first few rows of the uploaded file
        # st.write("### Uploaded File Preview")
        # st.write(df_test.head())

        # Load the training data (college_data.csv)
        df_train = load_data('college_data.csv')

        # Train the model and scaler on training data
        trained = train_model(df_train)

        if trained:
            scaler, model = trained

            # Preprocess and predict using uploaded file for testing
            ranked_colleges_names = preprocess_and_predict(df_test, scaler, model)

            # Display the top colleges with predicted rankings
            st.write("### Predicted College Rankings")
            st.write(ranked_colleges_names)

if __name__ == '__main__':
    main()
