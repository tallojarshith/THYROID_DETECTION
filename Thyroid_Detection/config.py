import pymongo
import os
import certifi

class EnvironmentVariable:
    def __init__(self):
        self.mongo_db_url = os.getenv("MONGO_DB_URL")
        self.HERUKO_API_KEY = os.getenv("HERUKO_API_KEY")
        self.HERUKO_APP_NAME = os.getenv("HERUKO_APP_NAME")

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
