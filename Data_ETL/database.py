
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid  import ObjectId
import os
from dotenv import load_dotenv
load_dotenv()

MongoURL = os.environ.get('MONGO')


def connect_to_db():
    client = AsyncIOMotorClient(MongoURL)
    db = client['ai_search']
    property_collection = db["user"]
    return property_collection

async def fetch_all_properties():
    property_collection = connect_to_db()
    res = property_collection.find({}, {'vectors': 0})

    return list(await res.to_list(length=100000))
