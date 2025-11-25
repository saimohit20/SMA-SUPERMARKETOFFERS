import pandas as pd
from qdrant_client import QdrantClient, models
import sys
import os
from dotenv import load_dotenv
from embedders.bert_embedder import bert_embed
from cleaning.helpers import clean_price, build_unique_key

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ingest_bert(df: pd.DataFrame, pincode: str):
   
    try:
        
        df["price"] = df["price"].apply(clean_price)
        df = df.dropna(subset=["product_name", "price"])
        df = df[df["product_name"].str.strip() != ""]
        df["unique_key"] = df.apply(build_unique_key, axis=1)
        

        # FETCH EXISTING BERT ITEMS
        existing = []
        offset = None

        while True:
            items, offset = qdrant.scroll(
                collection_name="offers_bert",
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

        # NEW + UPDATE
        new_rows = []
        update_ids = []

        for _, row in df.iterrows():
            key = row["unique_key"]
            if key in existing_map:
                if existing_map[key]["pincode"] != "ALL":
                    update_ids.append(existing_map[key]["id"])
            else:
                new_rows.append(row)


        # UPDATE
        if update_ids:
            qdrant.set_payload(
                collection_name="offers_bert",
                payload={"pincode": "ALL"},
                points=update_ids
            )

        # INGEST NEW ITEMS
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            new_df["pagecontent"] = new_df.apply(
                lambda r: f"{r['product_name']} at {r['store_name']} for {r['price']} EUR | category: {r['category']} | pincode: {pincode}",
                axis=1
            )

            texts = new_df["pagecontent"].tolist()
            print(f"Embedding {len(texts)} BERT vectors...")
            embeddings = bert_embed(texts)
            new_df["embedding"] = embeddings

            points = [
                models.PointStruct(
                    id=abs(hash(row["unique_key"] + "_bert")) % (10**12),
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

            qdrant.upsert("offers_bert", points)
            print(f"Upserted {len(points)} BERT items")

        print(" BERT ingestion DONE")

    except Exception as e:
        raise Exception(f"BERT ingestion failed: {e}")