import pandas as pd
from qdrant_client import QdrantClient, models
import google.generativeai as genai
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from cleaning.helpers import clean_price, build_unique_key

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
EMBED_MODEL = "text-embedding-004"
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# INGESTION INTO offers collection
def ingest_gemini(df: pd.DataFrame, pincode: str):
    
    try:
        
        df["price"] = df["price"].apply(clean_price)
        df = df.dropna(subset=["product_name", "price"])
        df = df[df["product_name"].str.strip() != ""]
        df["unique_key"] = df.apply(build_unique_key, axis=1)
       

        # FETCH EXISTING GEMINI ITEMS
        existing = []
        offset = None

        while True:
            items, offset = qdrant.scroll(
                collection_name="offers",
                limit=200,
                with_vectors=False,
                offset=offset
            )
            if not items:
                break
            existing.extend(items)
            if offset is None:
                break

        existing_map = {
            f"{pt.payload['product_name']}_{pt.payload['store_name']}_{pt.payload['price']}":
                {"id": pt.id, "pincode": pt.payload["pincode"]}
            for pt in existing
        }


        # SPLIT INTO NEW + UPDATE
        new_rows = []
        update_ids = []

        for _, row in df.iterrows():
            key = row["unique_key"]
            if key in existing_map:
                if existing_map[key]["pincode"] != "ALL":
                    update_ids.append(existing_map[key]["id"])
            else:
                new_rows.append(row)

        print(f"New items: {len(new_rows)}")
        print(f" Update to ALL: {len(update_ids)}")

  
        # UPDATE EXISTING TO ALL
        if update_ids:
            qdrant.set_payload(
                collection_name="offers",
                payload={"pincode": "ALL"},
                points=update_ids
            )
            
        # INGEST NEW ITEMS
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            new_df["pagecontent"] = new_df.apply(
                lambda r: f"{r['product_name']} at {r['store_name']} for {r['price']} EUR | category: {r['category']} | pincode: {pincode}",
                axis=1,
            )

            # EMBEDDINGS (Gemini)
            texts = new_df["pagecontent"].tolist()
            print(f"Embedding {len(texts)} Gemini vectors...")
            embeddings = genai.embed_content(
                    model=EMBED_MODEL,
                    content=texts
                )["embedding"]

            new_df["embedding"] = embeddings

            points = [
                models.PointStruct(
                    id=abs(hash(row["unique_key"])) % (10**12),
                    vector=row["embedding"],
                    payload={
                        "category": row["category"],
                        "product_name": row["product_name"],
                        "price": float(row["price"]),
                        "pincode": pincode,
                        "store_name": row["store_name"],
                        "product_url": row.get("product_url"),
                        "etl_version": 1,
                    }
                )
                for _, row in new_df.iterrows()
            ]

            qdrant.upsert("offers", points)
            print(f"Upserted {len(points)} Gemini items")

        print("Gemini ingestion DONE")

    except Exception as e:
        raise Exception(f"Gemini ingestion failed: {e}")