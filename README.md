# Supermarket Offers Assistant 🛒

A supermarket shopping assistant that helps you find the best deals from German supermarkets REWE and ALDI.

## What does this app do?

This app helps you:
- Find the best prices for products you want to buy.
- Compare deals from supermarkets.
- Get recommendations based on your location.
- The goal is to save time by not checking each store website manually.

## How it works

1. **Enter your location** - Type in your German postal code (5 digits)
2. **Ask what you want** - Type what product you're looking for (like "milk", "bread", "apples")
3. **Get recommendation** - The AI finds the best deals and shows you prices, stores, and links.


## Files explained

- `ui/` - The main app interface
- `supermarket_scrapers/` - Gets data from REWE and ALDI websites  
- `rag_engine/` - The simiratity search that finds sutiable products.
- `embedders/` - Helpers functions for embedding.
- `scraping_engine/` - data processing, embedding, data ingestion. 

## How to run

1. Install Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Start the app:
   ```
   streamlit run ui/ui.py
   ```

3. Open your web browser and go to the link shown


## Technology used

- **Streamlit** 
- **Selenium** 
- **Qdrant** 
- **Google Gemini & Qwen** 
- **BERT**
- **Pandas** 

---
