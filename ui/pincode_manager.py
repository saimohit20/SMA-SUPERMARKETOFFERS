from qdrant_client import QdrantClient, models
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Initialize Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)

# Function to update pincode registry 
def update_pincode_registry(pincode: str, num_products: int = None):
    
    try:
        # Check if pincodes collection exists, create if not
        collections = qdrant.get_collections()
        pincode_collection_exists = any(
            collection.name == "pincodes" 
            for collection in collections.collections
        )
        
        if not pincode_collection_exists:
            qdrant.create_collection(
                collection_name="pincodes",
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
            )
        
        # Prepare pincode data
        payload = {
            "pincode": pincode,
            "scraped_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        if num_products is not None:
            payload["num_products"] = num_products
            
        # Store in Qdrant
        qdrant.upsert(
            collection_name="pincodes",
            points=[
                models.PointStruct(
                    id=hash(pincode) % (2**63 - 1), 
                    vector=[0.0],  
                    payload=payload
                )
            ]
        )
        
        return True
        
    except Exception as e:
        print(f"Error updating pincode registry: {e}")
        return False

# Function to check if a pincode exists
def check_pincode_exists(pincode: str):
    
    try:
        found, _ = qdrant.scroll(
            collection_name="pincodes",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="pincode", match=models.MatchValue(value=pincode))]
            ),
            limit=1
        )
        
        return len(found) > 0
            
    except Exception as e:
        print(f"Error checking pincode: {e}")
        return False
