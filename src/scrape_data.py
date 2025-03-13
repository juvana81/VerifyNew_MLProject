import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL = "https://www.allsides.com/media-bias/media-bias-ratings"

def scrape_allsides():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")
    
    data = []
    table = soup.find("table", {"class": "views-table"})
    
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        source = cols[0].text.strip()
        bias = cols[1].text.strip()
        data.append([source, bias])
    
    df = pd.DataFrame(data, columns=["Source", "Bias"])
    
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/media_bias.csv", index=False)
    print("Data saved to data/media_bias.csv")

if __name__ == "__main__":
    scrape_allsides()
