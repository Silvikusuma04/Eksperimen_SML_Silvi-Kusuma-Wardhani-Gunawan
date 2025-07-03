import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import joblib

def main():
    try:
        df = pd.read_csv("padi.csv")
    except FileNotFoundError:
        print("Error: File 'padi.csv' tidak ditemukan.")
        return

    label_encoder = LabelEncoder()
    df['Provinsi'] = label_encoder.fit_transform(df['Provinsi'])

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    scaler = MinMaxScaler()
    df_normalized = df_filtered.copy()
    
    numeric_columns = df_normalized.select_dtypes(include=['int64', 'float64']).columns
    if 'Provinsi' in numeric_columns:
        numeric_columns = numeric_columns.drop('Provinsi')

    df_normalized[numeric_columns] = scaler.fit_transform(df_normalized[numeric_columns])

    output_dir = 'preprocessing/padi_preprocessing'
    os.makedirs(output_dir, exist_ok=True)

    processed_data_path = os.path.join(output_dir, 'preprocessing_padi.csv')
    df_normalized.to_csv(processed_data_path, index=False)

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    
    print(f"Preprocessing selesai. Output disimpan di {output_dir}")

if __name__ == '__main__':
    main()