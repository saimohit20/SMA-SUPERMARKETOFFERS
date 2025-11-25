import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_rewe(pincode: str):
    
    start_time = time.time()
    
    driver = webdriver.Chrome()
    driver.get("https://www.rewe.de/angebote/")
    wait = WebDriverWait(driver, 15)

    # Accept cookies
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            clicked = driver.execute_script("""
                const root = document.querySelector('#usercentrics-root');
                if (root && root.shadowRoot) {
                    const btn = root.shadowRoot.querySelector("button[data-testid='uc-accept-all-button']");
                    if (btn) { btn.click(); return true; }
                }
                return false;
            """)
            if clicked:
                print("Cookies accepted")
                break
            time.sleep(0.25)
    except Exception:
        pass

    time.sleep(2)

    try:
        wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.sos-category__content"))
        )
    except:
        pass

    records = []
    sections = driver.find_elements(By.CSS_SELECTOR, "div.sos-category__content")
    total_categories = len(sections)
    print(f"Total categories found: {total_categories}")

    for sec in sections:
        try:
            category = sec.find_element(
                By.CSS_SELECTOR,
                ".sos-category__content-title h2"
            ).text.strip()
        except:
            category = "Unknown"

        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sec)
        time.sleep(0.3)

        offers = sec.find_elements(By.CSS_SELECTOR, "div.sos-offer")
        
        for offer in offers:
            # name
            try:
                name = offer.find_element(
                    By.CSS_SELECTOR,
                    "a[data-testid='offer-title-link']"
                ).text.strip()
            except:
                name = ""

            # price
            try:
                price = offer.find_element(
                    By.CSS_SELECTOR,
                    ".cor-offer-price__tag-price"
                ).text.strip()
            except:
                price = ""

            # product URL 
            try:
                nan = offer.get_attribute("data-offer-nan")
                product_url = f"https://shop.rewe.de/p/{nan}/" if nan else ""
            except Exception:
                product_url = ""

            if name:
                records.append({
                    "category": category,
                    "product_name": name,
                    "price": price,
                    "product_url": product_url,  
                    "pincode": str(pincode),
                    "store_name": "REWE"
                })

    driver.quit()

    # Display scraping summary
    end_time = time.time()
    total_time = end_time - start_time
    total_products = len(records)
    
    print(f"Total products: {total_products}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time: {total_time/max(total_products, 1):.3f} seconds")

    return pd.DataFrame(records)