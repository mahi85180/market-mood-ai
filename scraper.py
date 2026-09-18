# scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime
from config import SITES

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}


def _site_file(site_name):
    safe = re.sub(r"[^a-zA-Z0-9]", "_", site_name)
    return os.path.join(DATA_DIR, f"{safe}.csv")


def fetch_page(url, timeout=15):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == 2:
                raise e
    return ""


def extract_rows(html):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if len(cells) >= 2:
                date = cells[0]
                numbers = []
                for cell in cells[1:]:
                    nums = re.findall(r"\d", cell)
                    numbers.extend(nums)
                if numbers and re.search(r"\d", date):
                    rows.append({"date": date, "numbers": numbers})
    if not rows:
        for line in soup.get_text("\n", strip=True).split("\n"):
            nums = re.findall(r"\d", line)
            if len(nums) >= 2 and len(line) < 100:
                rows.append({"date": line[:20], "numbers": nums})
    return rows


def load_history(site_name):
    path = _site_file(site_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["numbers"] = df["numbers"].apply(
            lambda x: x.split(",") if isinstance(x, str) and x else []
        )
        return df
    return pd.DataFrame(columns=["date", "numbers", "fetched_at"])


def save_history(site_name, df):
    df_copy = df.copy()
    df_copy["numbers"] = df_copy["numbers"].apply(lambda x: ",".join(x))
    df_copy.to_csv(_site_file(site_name), index=False)


def update_site(site_name):
    if site_name not in SITES:
        return None, f"Site '{site_name}' config me nahi hai."
    url = SITES[site_name]
    try:
        html = fetch_page(url)
        new_rows = extract_rows(html)
        if not new_rows:
            return None, f"{site_name}: koi data nahi mila."
        new_df = pd.DataFrame(new_rows)
        new_df["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        old = load_history(site_name)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined["key"] = combined["date"].astype(str) + "|" + combined["numbers"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x)
        )
        combined = combined.drop_duplicates(subset="key", keep="last").drop(columns="key")
        save_history(site_name, combined)
        return combined, f"{site_name}: {len(new_rows)} rows fetched, total {len(combined)}."
    except Exception as e:
        return None, f"{site_name} error: {e}"


def update_all_sites():
    results = {}
    for site in SITES:
        df, msg = update_site(site)
        results[site] = {"df": df, "msg": msg}
    return results
