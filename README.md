# Supermarket Offers Assistant 🛒

A smart shopping assistant that helps you find the best deals from German supermarkets REWE and ALDI using AI.

## What does this app do?

This app helps you:
- Find the best prices for products you want to buy.
- Compare deals between REWE and ALDI supermarkets.
- Get smart recommendations based on your location.
- Save time by not checking each store website manually.

## How it works

1. **Enter your location** - Type in your German postal code (5 digits)
2. **Ask what you want** - Type what product you're looking for (like "milk", "bread", "apples")
3. **Get smart answers** - The AI finds the best deals and shows you prices, stores, and links

## What's inside

The app uses three different AI models:
- **Gemini** - Google's AI model (default choice)
- **BERT** - Good for understanding what you really mean
- **Qwen** - Another smart AI option

## Main parts

- **Web scrapers** - Automatically get latest offers from REWE and ALDI websites
- **Smart search** - Uses AI to understand what you want and find matching products
- **Price comparison** - Shows you the cheapest options
- **Easy interface** - Simple chat-like experience

## Files explained

- `ui/` - The main app interface you see
- `supermarket_scrapers/` - Gets data from REWE and ALDI websites  
- `rag_engine/` - The smart search that finds what you need
- `embedders/` - Converts product info into numbers AI can understand
- `scraping_engine/` - Processes and stores the scraped data

## How to run

1. Install Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Start the app:
   ```
   streamlit run code/ui/ui.py
   ```

3. Open your web browser and go to the link shown



## Example questions you can ask

- "Show me cheap milk"
- "What fruits are on sale?"
- "Find bread under 2 euros"
- "Best pasta deals"
- "Organic vegetables near me"

## Important notes

- First time using a new postal code takes 30-60 seconds (getting fresh data)
- After that, searches are very fast
- The app remembers what it found so you don't wait again
- Works best with common grocery items

## Technology used

- **Streamlit** - Makes the web interface
- **Selenium** - Gets data from store websites
- **Qdrant** - Stores product information
- **Google Gemini** - Main AI for understanding and recommendations
- **BERT & Qwen** - Alternative AI models
- **Pandas** - Handles data processing

---

*This project helps German shoppers save money by finding the best supermarket deals using artificial intelligence.*
