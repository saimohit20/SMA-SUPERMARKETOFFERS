import json
import pandas as pd
from qdrant_client import QdrantClient, models
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from embedders.bert_embedder import bert_embed
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def perform_rag_bert(query: str, pincode: str) -> str:
  
    if not query:
        return json.dumps({"error": "Please enter a product-related query."})

    try:
        
        # Embed query with BERT
        qvec = bert_embed(query)[0]

        # Filter by pincode
        filter_cond = None
        if pincode != "ALL":
            filter_cond = models.Filter(
                must=[
                    models.FieldCondition(
                        key="pincode",
                        match=models.MatchAny(any=["ALL", pincode])
                    )
                ]
            )

        # Search in offers_bert
        hits = qdrant.search(
            collection_name="offers_bert",
            query_vector=qvec,
            query_filter=filter_cond,
            limit=5
        )

        docs = [h.payload for h in hits]
        if not docs:
            return json.dumps({"error": "No matching products found for your query."})


        # Optional: store for debugging like original code
        try:
            import streamlit as st
            df = pd.DataFrame(docs)
            st.session_state["last_top10_df_bert"] = df.copy()
        except Exception:
            pass

        # Pick top 2 by similarity (first 2 from vector search results)
        print("[BERT RAG] Retrieved product names:")
        for d in docs:
            print(f"  {d.get('product_name', '?')}")
        top2 = docs[:2]

        products_json = []
        for d in top2:
            products_json.append({
                "product_name": d["product_name"],
                "price": float(d["price"]),
                "store": d["store_name"],
                "product_url": d.get("product_url"),
                "pincode": d["pincode"]
            })
        

        recommendation = (
            "Based on BERT vector search, here are the most relevant "
            "products matching your query by semantic similarity."
        )

        response = {
            "products": products_json,
            "recommendation": recommendation
        }
        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during BERT product search: {str(e)}"
        })