from pymongo.mongo_client import MongoClient
import certifi
uri = "mongodb+srv://tallojarshith123:tallojarshith123@cluster0.rbcqhxv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri,tlsCAFile=certifi.where())
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)