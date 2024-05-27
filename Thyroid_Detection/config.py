import pymongo
import os
import certifi

class EnvironmentVariable:
    def __init__(self):
        self.mongo_db_url = os.getenv("MONGO_DB_URL")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_access_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Create an instance of EnvironmentVariable to access its attributes
env_var = EnvironmentVariable()

# Check if the MongoDB URL is available
if env_var.mongo_db_url:
    # Define the TLS/SSL options
    tls_options = {'ca_certs': certifi.where()}

    # Connect to MongoDB using PyMongo with TLS/SSL options
    mongo_client = pymongo.MongoClient(env_var.mongo_db_url, tls=True, tlsCAFile=tls_options['ca_certs'])
    # Now you can use mongo_client to interact with MongoDB

    # Define the target column
    TARGET_COLUMN = "Class"

    # Additional code for interacting with MongoDB can go here

else:
    print("MongoDB URL not provided in environment variables.")
