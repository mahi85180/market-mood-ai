# scrape_latest.py - Homepage se latest results uthao
import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

GAME_NAMES = [
    "SRIDEVI", "TIME BAZAR", "MADHUR DAY", "MILAN DAY",
    "RAJDHANI DAY", "SUPREME DAY", "KALYAN",
    "SRIDEVI NIGHT", "MADHUR NIGHT", "SUPREME NIGHT",
    "MILAN NIGHT", "KALYAN NIGHT", "RAJDHANI NIGHT", "MAIN BAZAR",
]


def fetch_latest_results():
    """Homepage se saare games ke latest results uthao"""
    url = "https://sattamatkano1.me/"
    r = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "lxml")

    results = {}

    # Get all text
    text = soup.get_text("\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for i, line in enumerate(lines):
        upper = line.upper().replace(" ", "")
        for game in GAME_NAMES:
            game_clean = game.upper().replace(" ", "")
            if game_clean in upper and len(line) < 30:
                # Search next 3 lines for pattern XXX-XX-XXX
                for j in range(1, 5):
                    if i + j >= len(lines):
                        break
                    candidate = lines[i + j]
                    match = re.match(r"^(\d{3})-(\d{2})-(\d{3})$", candidate.strip())
                    if match:
                        results[game] = {
                            "open_panna": match.group(1),
                            "jodi": match.group(2),
                            "close_panna": match.group(3),
                        }
                        break

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 HOMEPAGE LATEST RESULTS")
    print("=" * 70)

    results = fetch_latest_results()

    if not results:
        print("❌ Koi result nahi mila")
    else:
        print(f"\n✅ {len(results)} games ke results mile:\n")
        for game, r in results.items():
            open_digit = sum(int(d) for d in r["open_panna"]) % 10
            close_digit = sum(int(d) for d in r["close_panna"]) % 10
            print(f"📊 {game}")
            print(f"   {r['open_panna']}-{r['jodi']}-{r['close_panna']}")
            print(f"   Open={open_digit}  Jodi={r['jodi']}  Close={close_digit}")
            print()

    print("=" * 70)
