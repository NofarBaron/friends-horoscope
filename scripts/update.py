"""Fetch today's horoscope for each friend's sign, translate to Hebrew, write docs/data.json.

Source:      freehoroscopeapi.com (no key)
Translation: Helsinki-NLP/opus-mt-en-he (open model, runs on the GitHub runner CPU)
"""
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://freehoroscopeapi.com/api/v1/get-horoscope/daily?sign={sign}"
MODEL = "Helsinki-NLP/opus-mt-en-he"

# (sign, Hebrew name, symbol, start month/day) — sign starts on that day
SIGNS = [
    ("capricorn",   "גדי",    "♑", (12, 22)),
    ("aquarius",    "דלי",    "♒", (1, 20)),
    ("pisces",      "דגים",   "♓", (2, 19)),
    ("aries",       "טלה",    "♈", (3, 21)),
    ("taurus",      "שור",    "♉", (4, 20)),
    ("gemini",      "תאומים", "♊", (5, 21)),
    ("cancer",      "סרטן",   "♋", (6, 21)),
    ("leo",         "אריה",   "♌", (7, 23)),
    ("virgo",       "בתולה",  "♍", (8, 23)),
    ("libra",       "מאזניים", "♎", (9, 23)),
    ("scorpio",     "עקרב",   "♏", (10, 23)),
    ("sagittarius", "קשת",    "♐", (11, 22)),
]
RANGES = {
    "aries": "21.3–19.4", "taurus": "20.4–20.5", "gemini": "21.5–20.6",
    "cancer": "21.6–22.7", "leo": "23.7–22.8", "virgo": "23.8–22.9",
    "libra": "23.9–22.10", "scorpio": "23.10–21.11", "sagittarius": "22.11–21.12",
    "capricorn": "22.12–19.1", "aquarius": "20.1–18.2", "pisces": "19.2–20.3",
}


def sign_for(birthday: str) -> tuple:
    d = date.fromisoformat(birthday)
    md = (d.month, d.day)
    # Walk the year in order; the last start <= md wins (Capricorn wraps the year end).
    ordered = sorted(SIGNS, key=lambda s: s[3])
    current = ordered[-1]  # Capricorn covers Jan 1 until Aquarius starts
    for s in ordered:
        if md >= s[3]:
            current = s
    return current


def fetch(sign: str) -> dict:
    req = urllib.request.Request(API.format(sign=sign), headers={"User-Agent": "friends-horoscope"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def make_translator():
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL)

    def translate(text: str) -> str:
        # Sentence by sentence: Marian models degrade on long inputs.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        batch = tok(sentences, return_tensors="pt", padding=True, truncation=True)
        out = model.generate(**batch, num_beams=4, max_new_tokens=200)
        return " ".join(tok.batch_decode(out, skip_special_tokens=True))

    return translate


def main() -> int:
    friends = json.loads((ROOT / "friends.json").read_text(encoding="utf-8"))
    groups: dict[str, dict] = {}
    for f in friends:
        key, he, symbol, _ = sign_for(f["birthday"])
        g = groups.setdefault(key, {"sign": key, "sign_he": he, "symbol": symbol,
                                    "range": RANGES[key], "friends": []})
        g["friends"].append(f["name"])

    translate = make_translator()
    cards, api_date = [], None
    for key, g in groups.items():
        data = fetch(key)
        api_date = data.get("date", api_date)
        g["text_en"] = data["horoscope"]
        g["text_he"] = translate(data["horoscope"])
        cards.append(g)
        print(f"{key}: {', '.join(g['friends'])}", file=sys.stderr)

    order = [s[0] for s in sorted(SIGNS, key=lambda s: s[3])]
    cards.sort(key=lambda c: order.index(c["sign"]))
    out = {
        "date": api_date,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "model": MODEL,
        "cards": cards,
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
