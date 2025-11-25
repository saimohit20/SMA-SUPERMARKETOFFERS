#Convert price string to float. If conversion fails, returns None.
def clean_price(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.replace("€", "").replace(",", ".").strip())
    except:
        return None

# Dedupe key: same product_name + store_name + price.
def build_unique_key(row):
    return f"{row['product_name']}_{row['store_name']}_{row['price']}"
