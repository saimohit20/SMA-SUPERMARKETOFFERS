import streamlit as st
import json
import pandas as pd
from qdrant_client import QdrantClient, models
import os
import warnings
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# --------------------------------------------------
# Import engines 
# --------------------------------------------------
from supermarket_scrapers.rewe_scraper import scrape_rewe
from supermarket_scrapers.aldi_scraper import scrape_aldi
from scraping_engine.scraper_engine import ingest_gemini
from scraping_engine.bert_scraper_engine import ingest_bert
from scraping_engine.qwen_scraper_engine import ingest_qwen
from rag_engine.rag_engine import perform_rag
from rag_engine.bert_rag_engine import perform_rag_bert
from rag_engine.qwen_rag_engine import perform_rag_qwen
import google.generativeai as genai

# --------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

genai.configure(api_key=GENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# --------------------------------------------------
# Load CSS styles
# --------------------------------------------------
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# cleaning input
# --------------------------------------------------

def normalize_pincode(s: str):
    s = (s or "").strip().upper()
    if s == "" or s == "ALL":
        return "ALL"
    return "".join(ch for ch in s if ch.isdigit())


# --------------------------------------------------
# Post-Processing
# --------------------------------------------------

def format_rag_response(response_json: str):
    try:
        data = json.loads(response_json)
        if "error" in data:
            return f"Error: {data['error']}"

        recommendation = (data.get("recommendation") or "").strip()
        products = data.get("products", [])

        # Build a unified products table (only required fields)
        if not products:
            return recommendation or "No products found."

        def fmt_price(p):
            if "price" in p:
                try:
                    return f"€{float(p['price']):.2f}"
                except Exception:
                    return str(p['price'])
            return ""

        def fmt_link(p):
            url = p.get("product_url")
            return f"<a href='{url}'>View</a>" if url else ""

        rows_html = []
        for p in products:
            rows_html.append(
                f"<tr><td>{p.get('product_name','')}</td><td>{fmt_price(p)}</td><td>{p.get('store','')}</td><td>{fmt_link(p)}</td></tr>"
            )
        table_html = (
            "<table><thead><tr><th>Product</th><th>Price</th><th>Store</th><th>Link</th></tr></thead><tbody>"
            + "".join(rows_html) + "</tbody></table>"
        )

        parts = []
        if recommendation:
            parts.append(f"Recommendation:\n{recommendation}\n")
        parts.append(table_html)
        return "\n\n".join(parts).strip()
    except Exception:
        return response_json


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📍 Location")
    pincode_input = st.text_input("Enter pincode", value="ALL")
    pincode = normalize_pincode(pincode_input)

    if pincode != "ALL" and len(pincode) != 5:
        st.error("Pincode must be exactly 5 digits or ALL.")
        st.stop()

    st.success(f"{pincode}")

    st.markdown("---")

    st.markdown("## Model Selection")
    model_choice = st.radio("Choose model:", ["Gemini", "BERT", "Qwen"])

    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# --------------------------------------------------
# MAIN AREA (title + centered chat container)
# --------------------------------------------------

st.title("🛒 Supermarket Offers Assistant")
st.markdown("Find the best deals from REWE and ALDI using AI.")
st.markdown("---")

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------

ss = st.session_state

if "messages" not in ss:
    ss["messages"] = []

if "processing" not in ss:
    ss["processing"] = False

if "scraping_phase" not in ss:
    ss["scraping_phase"] = None

if "loading_message" not in ss:
    ss["loading_message"] = ""


# --------------------------------------------------
# WELCOME MESSAGE (when no conversation started)
# --------------------------------------------------

if not ss["messages"] and not ss.get("processing"):
    st.markdown(
        "<div style='text-align: center; padding: 50px 0;'><h3>Welcome! 👋</h3></div>", 
        unsafe_allow_html=True
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for msg in ss["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").markdown(
            format_rag_response(msg["content"]), unsafe_allow_html=True
        )


if ss.get("processing"):
    with st.chat_message("assistant"):
        st.markdown(
            f"""
            <div class='typing'>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
            </div>
            <div style="margin-top:10px;">{ss.get("loading_message", "Working...")}</div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

query = st.chat_input("Ask about supermarket offers...")


# --------------------------------------------------
# CHAT SEND HANDLING
# --------------------------------------------------

if query:
    ss["messages"].append({"role": "user", "content": query})
    ss["processing"] = True
    ss["scraping_phase"] = "checking"
    ss["current_query"] = query
    ss["current_pincode"] = pincode
    ss["current_model"] = model_choice
    ss["loading_message"] = "Checking available offers..."
    st.rerun()


# --------------------------------------------------
# PROCESSING PIPELINE
# --------------------------------------------------

if ss.get("processing"):
    q = ss["current_query"]
    pin = ss["current_pincode"]
    model = ss["current_model"]
    
    # Debug logging
    print(f"[UI] Processing phase: {ss.get('scraping_phase')} | Query: {q} | Pin: {pin} | Model: {model}")

    # -------- PHASE 1 — CHECK DATA --------
    if ss["scraping_phase"] == "checking":
        print(f"[UI] Checking for existing data for pincode: {pin}")
        from pincode_manager import check_pincode_exists
        
        if pin == "ALL" or check_pincode_exists(pin):
            print(f"[UI] Data found for pincode {pin}, moving to RAG phase")
            ss["scraping_phase"] = "rag"
            ss["loading_message"] = "Fetching existing offers..."
        else:
            print(f"[UI] No data found for pincode {pin}, starting scraping")
            ss["scraping_phase"] = "scraping"
            ss["loading_message"] = "Scraping REWE & ALDI… Please wait (40–70 seconds)."
        st.rerun()

    # -------- PHASE 2 — SCRAPE --------
    if ss["scraping_phase"] == "scraping":
        try:
            print(f"[UI] Starting scraping for pincode {pin}")
            ss["loading_message"] = "Scraping REWE data..."

            df_rewe = scrape_rewe(pin)
            print(f"[UI] REWE scraping complete: {len(df_rewe)} items")

            ss["loading_message"] = "Scraping ALDI data..."
            df_aldi = scrape_aldi(pin)
            print(f"[UI] ALDI scraping complete: {len(df_aldi)} items")

            df = pd.concat([df_rewe, df_aldi], ignore_index=True)
            print(f"[UI] Combined data: {len(df)} total items")
            ss["loading_message"] = "Processing and indexing data..."
            
            ingest_gemini(df, pin)
            print(f"[UI] Gemini ingestion complete")

            ingest_bert(df, pin)
            print(f"[UI] BERT ingestion complete")

            ingest_qwen(df, pin)
            print(f"[UI] Qwen ingestion complete")

            from pincode_manager import update_pincode_registry
            update_pincode_registry(pin)
            print(f"[UI] Pincode registry updated for {pin}")
            ss["scraping_phase"] = "rag"
            ss["loading_message"] = "Searching best matches…"
            print(f"[UI] Scraping complete, moving to RAG phase")
            st.rerun()

        except Exception as e:
            print(f"[UI] Error during scraping: {str(e)}")
            st.error(f"Error during scraping: {str(e)}")
            ss["processing"] = False
            ss["scraping_phase"] = None
            ss["loading_message"] = ""
            st.rerun()

    # -------- PHASE 3 — RAG --------
    if ss["scraping_phase"] == "rag":
        try:
            if model == "Gemini":
                result = perform_rag(q, pin)
            elif model == "BERT":
                result = perform_rag_bert(q, pin)
            else:
                result = perform_rag_qwen(q, pin)
            ss["messages"].append({"role": "assistant", "content": result})
            ss["processing"] = False
            ss["scraping_phase"] = None
            ss["loading_message"] = ""
            for k in ["current_query", "current_pincode", "current_model"]:
                if k in ss:
                    del ss[k]
            st.rerun()

        except Exception as e:
            st.error(f"Error during RAG search: {str(e)}")
            ss["processing"] = False
            ss["scraping_phase"] = None
            ss["loading_message"] = ""
            st.rerun()