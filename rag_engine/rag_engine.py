import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

genai.configure(api_key=GENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"


def extract_json(text: str) -> str:
    
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 1)
        text = parts[1] if len(parts) > 1 else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_search_query(original_query: str) -> str:

    try:
        prompt = f"""
                You are a supermarket product query extractor.
                Extract ONLY the concrete product items the user wants to purchase or compare.
                Output Rules:
                - Return ONLY a single line of comma-separated product search terms. No explanations.
                - Preserve essential modifiers: flavor (chocolate), type (whole), brand (if explicitly stated), form (fresh, frozen), key size if critical (e.g. 1kg rice) else omit.
                - Singularize plural forms (bananas -> banana, ice creams -> ice cream).
                - Merge flavor + product: chocolate ice cream, vanilla yogurt.
                - Remove conversational filler (I want, I like, please, can you, etc.).
                - If multiple distinct products: apple, banana, orange.
                - Order terms as they appear in the original request.
                - No duplicates; lowercase except brand capitalization if present.
                - If no clear product appears, return the original query unchanged.

                Examples:
                User: I want banana and apple -> banana, apple
                User: I like to eat ice creams, I like chocolate flavor -> chocolate ice cream
                User: Need fresh spinach and cherry tomatoes -> fresh spinach, cherry tomatoes
                User: Show me deals on Coca Cola 1.5L and Pepsi Max -> Coca Cola 1.5L, Pepsi Max
                User: Looking for cheap cereals -> cereal
                User request: "{original_query}"
                Final comma-separated product terms:
                """
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(prompt)
        refined_raw = resp.text.strip().strip('"')
        
        if not refined_raw:
            return original_query
        first_line = refined_raw.splitlines()[0]
        items = [i.strip() for i in first_line.split(',') if i.strip()]
        if not items:
            return original_query
        refined_single_line = ', '.join(items)
        print(f"[RAG] LLM refined search query: {refined_single_line}")
        return refined_single_line
    except Exception as e:
        print(f"[RAG] Query refinement failed, using original. Error: {e}")
        return original_query


def perform_rag(query: str, pincode: str) -> str:
   
    # Step 1: basic validation
    if not query:
        return json.dumps({"error": "Please enter a product-related query."})

    try:
        # Step 2: refine query and split into individual requested items
        refined_query = generate_search_query(query)
        requested_items = [t.strip() for t in refined_query.split(',') if t.strip()] or [refined_query]

        # Step 3: build optional pincode filter
        filter_cond = None
        if pincode != "ALL":
            filter_cond = models.Filter(must=[models.FieldCondition(
                key="pincode", match=models.MatchAny(any=["ALL", pincode])
            )])

        # Step 4: vector search per requested item (collect top 4 payloads)
        per_item_candidates = {}
        for item in requested_items:
            try:
                emb = genai.embed_content(model=EMBED_MODEL, content=item)["embedding"]
                hits = qdrant.search(
                    collection_name="offers",
                    query_vector=emb,
                    query_filter=filter_cond,
                    limit=4
                )
                per_item_candidates[item] = [h.payload for h in hits]
            except Exception as se:
                print(f"[RAG] Search failed for '{item}': {se}")
                per_item_candidates[item] = []

        # Debug: print only product names retrieved per requested item
        print("[RAG] GEMINI Retrieved product names:")
        for item, cands in per_item_candidates.items():
            names = [c.get('product_name', '?') for c in cands]
            if names:
                print(f"  {item}: {', '.join(names)}")
            else:
                print(f"  {item}: none")

        # Step 5: construct textual context grouping candidates per item for LLM 
        context_sections = []
        for item, cands in per_item_candidates.items():
            if not cands:
                context_sections.append(f"Requested item: {item}\n  (No candidates found)\n")
                continue
            lines = [f"Requested item: {item}"]
            for i, d in enumerate(cands):
                lines.append(
                    f"Candidate {i+1}: {d['product_name']} | Store: {d['store_name']} | Price: €{float(d['price']):.2f} | Category: {d.get('category','')} | URL: {d.get('product_url','N/A')}"
                )
            context_sections.append("\n".join(lines))
        context = "\n\n".join(context_sections)

        # Step 6: build LLM prompt for selecting one best product per item
        prompt = f"""
                    You are a supermarket shopping assistant. For each requested item, pick the SINGLE best matching product (at most one per item). If no suitable product exists for an item, skip it and later mention it as not found in the recommendation.

                    User original query: "{query}"
                    LLM refined item list: {requested_items}

                    Candidate products grouped by requested item:
                    {context}

                    Selection rules:
                    1. Choose at most one product per requested item.
                    2. Relevance first: exact or close semantic match to the requested item (including flavor/type/brand modifiers).
                    3. If multiple equally relevant, prefer cheaper price.
                    4. If no relevant candidate for an item, do NOT fabricate; simply omit from products list.
                    5. Recommendation: MAX 3 SHORT sentences, plain simple tone.
                    - For each chosen product: say why it was picked (cheapest, flavor/brand match, better size/value).
                    - Optionally mention ONE alternative you did not choose and why (e.g. "Also at REWE for €2.10 but higher price").
                    - If any requested items missing, last sentence: "onion not found" (list them).
                    - Avoid generic openings like "Here are"; go straight to the reasoning.

                    Return ONLY valid JSON with this format (variable length products array):
                    {{
                      "products": [
                        {{
                          "product_name": "exact matched product name",
                          "price": 0,
                          "store": "store name",
                          "product_url": "product URL or null",
                          "pincode": "product pincode"
                        }}
                      ],
                      "recommendation": "Up to 3 short sentences with reasoning as above."
                    }}

                    IMPORTANT:
                    - Do not include products for items with no match.
                    - Do not add explanatory text outside JSON.
                    """

        # Step 7: call LLM and attempt to parse JSON
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        print(f"LLM: Raw response: {resp.text}")

        try:
            cleaned = extract_json(resp.text)
            json_response = json.loads(cleaned)
            print(" [RAG] JSON parsed successfully")
            return json.dumps(json_response, indent=2)
        except json.JSONDecodeError:
            print("[RAG] JSON parse failed")

    # Step 8: outer error handler
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during product search: {str(e)}"
        })