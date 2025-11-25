import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_aldi(pincode: str):
    """Scrape ALDI products for a given pincode."""
    start_time = time.time()
    
    driver = webdriver.Chrome()
    driver.get("https://www.aldi-nord.de/filialen-und-oeffnungszeiten.html")

    # Accept cookies
    deadline = time.time() + 10
    clicked = False
    while time.time() < deadline and not clicked:
        clicked = driver.execute_script("""
            const root = document.querySelector('#usercentrics-root');
            if (root && root.shadowRoot) {
                const btn = root.shadowRoot.querySelector("button[data-testid='uc-accept-all-button']");
                if (btn) { btn.click(); return true; }
            }
            return false;
        """)
        if not clicked:
            time.sleep(0.25)
    print("Cookies accepted")

    # Enter pincode
    try:
        search_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "autocomplete-input"))
        )
        search_input.clear()
        search_input.send_keys(pincode)
        time.sleep(1)

        first_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "ul#autocomplete-dropdown li:first-child"))
        )
        driver.execute_script("arguments[0].click();", first_option)
        print(f"Entered pincode: {pincode}")
        time.sleep(2) 
    except Exception as e:
        print("Could not select location:", e)

    # Click 'ANGEBOTE'
    try:
        angebote_btn = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(@class, 'ubsf_location-list-item-cta') and contains(., 'ANGEBOTE')]")
            )
        )
        driver.execute_script("arguments[0].click();", angebote_btn)
        print("Clicked Angebote")
        time.sleep(2)  
    except Exception as e:
        print("Could not click 'ANGEBOTE':", e)

    # Scrape offers
    records = []
    categories = driver.find_elements(By.CSS_SELECTOR, "div.mod-tile-group")
    total_categories = len(categories)
    print(f"Total categories found: {total_categories}")

    for cat in categories:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", cat)
        time.sleep(0.5)

        try:
            category = cat.find_element(By.CSS_SELECTOR, "div.mod-headline h2").text.strip()
        except:
            category = "Unknown"

        products = cat.find_elements(By.CSS_SELECTOR, "div[data-t-name='ArticleTile']")
        
        for prod in products:
            data_raw = prod.get_attribute("data-article")
            if not data_raw:
                continue
            
            # Get product URL
            try:
                link_el = prod.find_element(By.CSS_SELECTOR, "a.mod-article-tile__action")
                href = link_el.get_attribute("href") or ""
                if href.startswith("/"):
                    product_url = "https://www.aldi-nord.de" + href
                else:
                    product_url = href
            except Exception:
                product_url = ""

            try:
                data_json = json.loads(data_raw.replace("&quot;", '"'))
                info = data_json.get("productInfo", {})
                records.append({
                    "category": category,
                    "product_name": info.get("productName"),
                    "price": info.get("priceWithTax"),
                    "product_url": product_url,       
                    "pincode": str(pincode),
                    "store_name": "ALDI"
                })
            except json.JSONDecodeError:
                continue

    driver.quit()
    
    # Display scraping summary
    end_time = time.time()
    total_time = end_time - start_time
    total_products = len(records)
    
    print(f"Total products: {total_products}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time: {total_time/max(total_products, 1):.3f} seconds")
    
    return pd.DataFrame(records)