import json
import ollama
import re
import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from embedders.qwen_embedder import embed_one

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
QWEN_GEN_MODEL = "qwen3:4b"

def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 1)
        text = parts[1] if len(parts) > 1 else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()

def generate_search_query_qwen(original_query: str) -> str:
    try:

        prompt = f"""
                You extract grocery product search terms.
                Return ONLY one line: comma-separated product items (with essential modifiers: flavor, brand if stated, form like fresh/frozen, size only if critical).
                Singularize plurals (bananas -> banana). Merge flavor + product (chocolate ice cream).
                Remove filler (I want, please, can you).
                Order terms as they appear. No duplicates. Keep brand capitalization.
                If no clear product terms, just return the original query.
                Examples:
                I want banana and apple -> banana, apple
                I like chocolate ice creams -> chocolate ice cream
                Need organic whole milk and unsalted butter -> organic whole milk, unsalted butter
                Show deals on Coca Cola 1.5L and Pepsi Max -> Coca Cola 1.5L, Pepsi Max
                Looking for cheap cereals -> cereal
                User request: "{original_query}"
                Products:
                """
        
        resp = ollama.generate(model=QWEN_GEN_MODEL, prompt=prompt, options={"temperature": 0.2})
        raw = resp.get("response", "").strip().strip('"')

        if not raw:
            return original_query
        first_line = raw.splitlines()[0]
        items = [i.strip() for i in first_line.split(',') if i.strip()]
        if not items:
            return original_query
        refined = ', '.join(items)
        print(f"[QWEN RAG] Refined items: {refined}")
        return refined
    
    except Exception as e:
        print(f"[QWEN RAG] Query refinement failed: {e}")
        return original_query

def perform_rag_qwen(query: str, pincode: str) -> str:

    # Step 1: validate
    if not query:
        return json.dumps({"error": "Please enter a product-related query."})
    try:
        
        # Step 2: refine and split requested items
        refined_query = generate_search_query_qwen(query)
        requested_items = [t.strip() for t in refined_query.split(',') if t.strip()] or [refined_query]


        # Step 3: optional pincode filter
        filter_cond = None
        if pincode != "ALL":
            filter_cond = models.Filter(must=[models.FieldCondition(key="pincode", match=models.MatchAny(any=["ALL", pincode]))])

        # Step 4: vector search per item
        per_item_candidates = {}
        for item in requested_items:
            try:
                emb = embed_one(item)
                hits = qdrant.search(collection_name="offers_qwen", query_vector=emb, query_filter=filter_cond, limit=4)
                per_item_candidates[item] = [h.payload for h in hits]
            except Exception as se:
                print(f"[QWEN RAG] Search failed for '{item}': {se}")
                per_item_candidates[item] = []


        # print only product names
        print("[QWEN RAG] Retrieved product names:")
        for item, cands in per_item_candidates.items():
            names = [c.get('product_name', '?') for c in cands]
            print(f"  {item}: {', '.join(names) if names else 'none'}")


        # Step 5: build context text
        sections = []
        for item, cands in per_item_candidates.items():
            if not cands:
                sections.append(f"Requested item: {item}\n  (No candidates found)\n")
                continue
            lines = [f"Requested item: {item}"]
            for i, d in enumerate(cands):
                lines.append(f"Candidate {i+1}: {d['product_name']} | Store: {d['store_name']} | Price: €{float(d['price']):.2f} | Category: {d.get('category','')} | URL: {d.get('product_url','N/A')}")
            sections.append("\n".join(lines))
        context = "\n\n".join(sections)


        # Step 6: selection prompt
        selection_prompt = f"""
                        Pick ONE best product per requested item. Skip items with no suitable match.
                        Original query: "{query}"
                        Refined items: {requested_items}

                        Candidates:
                        {context}

                        Rules:
                        1. Max one product per item.
                        2. Must semantically match item (consider flavor/brand/modifier).
                        3. Tie -> choose cheaper.
                        4. Do not fabricate missing products.
                        5. Recommendation: up to 3 short sentences, simple tone.
                        - State why each chosen product was selected (cheapest, brand match, better value).
                        - Optional: mention one pricier alternative not chosen ("Also at REWE for €2.10 but higher").
                        - Last sentence lists any missing items like: onion not found.
                        - Do NOT start with generic phrases.
                        Return ONLY JSON:
                        {{
                          "products": [
                            {{"product_name": "name", "price": 0, "store": "store", "product_url": "url or null", "pincode": "pincode"}}
                          ],
                          "recommendation": "Up to 3 short sentences as described."
                        }}
                        IMPORTANT: Strict JSON only.
                        """
        
        # Step 7: call model and parse JSON
        response = ollama.generate(model=QWEN_GEN_MODEL, prompt=selection_prompt, options={"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.1})
        raw = response.get("response", "").strip()
        print(f"[QWEN RAG] Raw response: {raw[:200]}")
        try:
            cleaned = extract_json(raw)
            if not cleaned.startswith('{'):
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    cleaned = m.group(0)
            data = json.loads(cleaned)
            print("[QWEN RAG] JSON parsed successfully")
            return json.dumps(data, indent=2)
        except Exception as e:
            print(f"[QWEN RAG] JSON parse failed: {e}")
    except Exception as e:
        print(f"[QWEN RAG] Error: {e}")
        return json.dumps({"success": False, "error": f"Error during Qwen product search: {str(e)}"})