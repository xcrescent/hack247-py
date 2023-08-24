import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

url = "https://www.amazon.in/gp/bestsellers/electronics/ref=zg_bs_nav_0"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Extract relevant data using BeautifulSoup
# For example, extracting product names and prices
product_names = [item.get_text() for item in soup.find_all("div", class_="product-name")]
prices = [item.get_text() for item in soup.find_all("span", class_="price")]

# Create a DataFrame from the fetched data
data = {
    "Product": product_names,
    "Price": prices
}
print(data)

if len(product_names) != len(prices):
    print("Error: Number of products and prices do not match.")
    exit(1)

elif len(product_names) == 0:
    print("Error: No products found.")
    exit(1)
    
df = pd.DataFrame(data)

# Perform basic data analysis
average_price = df["Price"].mean()
highest_price_product = df[df["Price"] == df["Price"].max()]["Product"].iloc[0]
lowest_price_product = df[df["Price"] == df["Price"].min()]["Product"].iloc[0]

# Create a bar chart of product prices
plt.bar(df["Product"], df["Price"])
plt.xlabel("Product")
plt.ylabel("Price")
plt.title("Product Prices")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save or display the chart
plt.savefig("product_prices.png")
plt.show()
