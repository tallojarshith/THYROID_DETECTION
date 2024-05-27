import pandas as pd
import json
import pymongo
import certifi

# Absolute path to your CSV file
DATA_FILE_PATH = r"C:\Users\Harshith\project-batch1\THYROID_DETECTION\notebook\hypothyroid_data.csv"
# MongoDB connection details
CONNECTION_URL = "mongodb+srv://tallojarshith123:tallojarshith123@cluster0.rbcqhxv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "HealthCare"
COLLECTION_NAME = "thyroid_data"

if __name__=="__main__":
    # Read CSV file into DataFrame
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    # Convert DataFrame to JSON
    df.reset_index(drop=True, inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # Setup MongoDB client and insert data
    client = pymongo.MongoClient(CONNECTION_URL, tlsCAFile=certifi.where())
    data_base = client[DATABASE_NAME]
    collection = data_base[COLLECTION_NAME]
    collection.insert_many(json_record)