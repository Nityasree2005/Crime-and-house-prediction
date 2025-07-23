# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_crime_model():
    try:
        # Try to load the Excel file instead of CSV
        # First try crp.xlsx
        try:
            print("Attempting to load crp.xlsx...")
            df = pd.read_excel('data/crp.xlsx')
            print("Successfully loaded crp.xlsx")
        except:
            # If that fails, try new_dataset.xlsx
            print("Attempting to load new_dataset.xlsx...")
            df = pd.read_excel('data/new_dataset.xlsx')
            print("Successfully loaded new_dataset.xlsx")
        
        # Print column names to verify
        print("Available columns:", df.columns.tolist())
        
        # Check if the target column exists
        if 'ViolentCrimesPerPop' not in df.columns:
            # If the target column doesn't exist, look for similar columns
            possible_target_columns = [col for col in df.columns if 'crime' in col.lower() or 'violent' in col.lower()]
            if possible_target_columns:
                target_column = possible_target_columns[0]
                print(f"Target column 'ViolentCrimesPerPop' not found. Using '{target_column}' instead.")
            else:
                # If no appropriate column found, use the last column as target (common convention)
                target_column = df.columns[-1]
                print(f"No crime-related target column found. Using '{target_column}' as target.")
        else:
            target_column = 'ViolentCrimesPerPop'
        
        # Select features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle non-numeric columns
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        X = X[numeric_columns]  # Keep only numeric columns
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model (RandomForest as an example)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model and scaler
        with open('models/crime_rate_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/crime_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names for later reference
        with open('models/crime_features.pkl', 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        print("Crime rate model trained and saved successfully.")
        
        # Optional: Evaluate the model
        score = model.score(X_test_scaled, y_test)
        print(f"Model R² score: {score:.4f}")
        
    except Exception as e:
        print(f"Error training crime model: {e}")
        import traceback
        traceback.print_exc()

def train_house_price_model():
    try:
        # Try different possible filenames for the Chennai house price data
        try:
            print("Attempting to load chennai_house_data.csv...")
            df = pd.read_csv('data/chennai_house_data.csv')
            print("Successfully loaded chennai_house_data.csv")
        except:
            try:
                print("Attempting to load Chennai.csv...")
                df = pd.read_csv('data/Chennai.csv')
                print("Successfully loaded Chennai.csv")
            except:
                try:
                    print("Attempting to load Chennai housing data.csv...")
                    df = pd.read_csv('data/Chennai housing data.csv')
                    print("Successfully loaded Chennai housing data.csv")
                except:
                    # Try to load Excel files if CSV not found
                    try:
                        print("Attempting to load Chennai.xlsx...")
                        df = pd.read_excel('data/Chennai.xlsx')
                        print("Successfully loaded Chennai.xlsx")
                    except:
                        raise Exception("Could not find Chennai house price dataset")
        
        # Print column names to verify
        print("Available columns:", df.columns.tolist())
        
        # Look for price column
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        if price_columns:
            price_column = price_columns[0]
        else:
            # If no price column found, look for other possible names
            possible_price_columns = [col for col in df.columns if any(term in col.lower() for term in ['cost', 'value', 'amount', 'rs'])]
            if possible_price_columns:
                price_column = possible_price_columns[0]
            else:
                # If still not found, use the last column (common convention)
                price_column = df.columns[-1]
        
        print(f"Using '{price_column}' as target column for house prices")
        
        # Look for location column
        location_columns = [col for col in df.columns if any(term in col.lower() for term in ['location', 'area', 'place', 'zone'])]
        if location_columns:
            location_column = location_columns[0]
            print(f"Using '{location_column}' for location")
            # One-hot encode the location
            df_encoded = pd.get_dummies(df, columns=[location_column])
        else:
            print("No location column found, skipping one-hot encoding")
            df_encoded = df
        
        # Select features and target
        X = df_encoded.drop(price_column, axis=1)
        y = df_encoded[price_column]
        
        # Handle non-numeric columns (convert categorical to numeric or drop)
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        X = X[numeric_columns]  # Keep only numeric columns
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model (RandomForest as an example)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model and scaler
        with open('models/chennai_house_price_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/house_price_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names for later reference
        with open('models/house_price_features.pkl', 'wb') as f:
            pickle.dump(list(X.columns), f)
        
        print("House price model trained and saved successfully.")
        
        # Optional: Evaluate the model
        score = model.score(X_test_scaled, y_test)
        print(f"Model R² score: {score:.4f}")
        
    except Exception as e:
        print(f"Error training house price model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Training Crime Rate Model...")
    train_crime_model()
    
    print("\nTraining Chennai House Price Model...")
    train_house_price_model()