import json
import re
from sentence_transformers import SentenceTransformer, util
import fasttext

INPUT_FILE = "data/raw.json"
OUTPUT_FILE = "data/processed.json"

# ==============================
# CATEGORIAS (DESC + KEYWORDS + NEGATIVE)
# ==============================

CATEGORY_MAP = {

    "crypto": {
        "desc": (
            "post about cryptocurrency markets, bitcoin, ethereum, altcoins, trading, "
            "blockchain adoption, tokens and digital assets"
        ),
        "keywords": ["crypto", "bitcoin", "ethereum", "btc", "eth", "altcoin", "trading", "token"],
        "negative": []
    },

    "leo": {
        "desc": (
            "post about leofinance, inleo platform, leo token, leo power and leo ecosystem"
        ),
        "keywords": ["leo", "leofinance", "inleo", "leodex", "leo power"],
        "negative": []
    },

    "hive": {
        "desc": (
            "post about hive blockchain, hive blog, hive community and hive ecosystem"
        ),
        "keywords": ["hive", "hive blockchain", "hive power", "hive blog", "hp"],
        "negative": []
    },

    "hive-engine": {
        "desc": (
            "post about hive-engine tokens, tribaldex, diesel pools and second layer tokens"
        ),
        "keywords": ["hive-engine", "tribaldex", "diesel pool", "swap.hive", "layer 2"],
        "negative": []
    },

    "defi": {
        "desc": (
            "post about decentralized finance, staking, yield farming and liquidity pools"
        ),
        "keywords": ["defi", "staking", "yield", "liquidity", "apy", "apr"],
        "negative": []
    },

    "nft": {
        "desc": (
            "post about nft collections, minting, digital art and blockchain collectibles"
        ),
        "keywords": ["nft", "mint", "collection", "floor price", "opensea"],
        "negative": []
    },

    "blockchain gaming": {
        "desc": (
            "post about blockchain games like splinterlands, play to earn and crypto gaming"
        ),
        "keywords": ["splinterlands", "sps", "p2e", "play to earn", "nft game", "ambush ability", "arcane foil", "epic elf", "life element", "attacks per round", "double strike", "spellbook", "cards", "strategies", "rulesets", "battle", "weak magic", "monsters", "mana cost", "summoner", "ranged monsters", "card collection", "players", "maxed cards"],
        "negative": []
    },

    "programming": {
        "desc": (
            "post about programming, coding, backend, frontend and software development"
        ),
        "keywords": ["code", "programming", "python", "javascript", "api", "backend"],
        "negative": []
    },

    "ai": {
        "desc": (
            "post about artificial intelligence, machine learning, neural networks, "
            "chatgpt, gemini, grok, deepseek"
        ),
        "keywords": ["ai", "chatgpt", "gpt", "gemini", "grok", "deepseek", "llm", "ai influencer", "generate","artificial", "ai"],
        "negative": []
    },

    "cybersecurity": {
        "desc": (
            "post about hacking, cybersecurity, exploits, privacy and security topics"
        ),
        "keywords": ["hacking", "cybersecurity", "exploit", "privacy", "security"],
        "negative": []
    },

    "gaming": {
        "desc": (
            "post about video games, gameplay and gaming experiences"
        ),
        "keywords": ["game", "gaming", "gameplay", "pc gaming", "console", "playing", "horror game", "the game", "playthrough", "steam", "epic games", "xbox", "playstation"],
        "negative": []
    },

    "finance": {
        "desc": (
            "post about finance, investing, money, stock market and business"
        ),
        "keywords": ["finance", "investment", "money", "profit", "income"],
        "negative": []
    },

    "entrepreneurship": {
        "desc": (
            "post about startups, business ideas, product building and making money"
        ),
        "keywords": ["startup", "business", "entrepreneur", "saas", "revenue"],
        "negative": []
    },

    "marketing": {
        "desc": (
            "post about digital marketing, seo, branding and content creation"
        ),
        "keywords": ["marketing", "seo", "branding", "content", "growth"],
        "negative": []
    },

    "art": {
        "desc": (
            "post about digital art, drawing, painting and creative work"
        ),
        "keywords": ["art", "drawing", "painting", "illustration", "design"],
        "negative": ["attack power", "recharge ability", "crafting"]
    },

    "daily report": {
        "desc": (
            "post about daily report curation"
        ),
        "keywords": ["daily", "q𝐮𝐚𝐥𝐢𝐭𝐲 posts of the day", "post link", "report", "curation"],
        "negative": ["actifit"]
    },

    "photography": {
        "desc": (
            "post about photography, camera work and visual storytelling"
        ),
        "keywords": ["photo", "photography", "camera", "portrait", "landscape", "mobile photography","technique", "liketu"],
        "negative": []
    },

    "writing": {
        "desc": (
            "post about writing, storytelling, essays and creative content"
        ),
        "keywords": ["writing", "story", "essay", "fiction", "poetry"],
        "negative": []
    },

    "music": {
        "desc": (
            "post about music, songs, production and sound creation"
        ),
        "keywords": ["music", "song", "beat", "album", "producer", "bts", "arirang"],
        "negative": []
    },

    "video": {
        "desc": (
            "post about video creation, youtube, editing, streaming and content production"
        ),
        "keywords": ["video", "youtube", "editing", "stream", "content creator"],
        "negative": []
    },

    "lifestyle": {
        "desc": (
            "post about daily life, personal experiences and motivation"
        ),
        "keywords": ["life", "daily", "experience", "lifestyle", "story"],
        "negative": []
    },

    "opinion": {
        "desc": (
            "post expressing personal opinion, thoughts and perspectives"
        ),
        "keywords": ["i think", "i believe", "opinion", "my view"],
        "negative": []
    },

    "travel": {
        "desc": (
            "post about travel, tourism and exploring places"
        ),
        "keywords": ["travel", "trip", "tourism", "vacation", "adventure"],
        "negative": []
    },

    "food": {
        "desc": (
            "post about food, cooking, recipes and meals"
        ),
        "keywords": ["food", "recipe", "cooking", "meal", "restaurant"],
        "negative": []
    },

    "fashion": {
        "desc": (
            "post about fashion, clothing and style"
        ),
        "keywords": ["fashion", "outfit", "style", "clothes", "brand"],
        "negative": []
    },

    "fitness": {
        "desc": (
            "post about fitness, workouts, health and exercise"
        ),
        "keywords": ["fitness", "gym", "workout", "exercise", "health", "took a walk", "walk over", "steps", "walking", "actifit", "run", "runner", "marathon", "race", "powerhiking", "raceday"],
        "negative": []
    },

    "sports": {
        "desc": (
            "post about sports, teams, matches and competitions"
        ),
        "keywords": ["sports", "match", "team", "score", "league"],
        "negative": []
    },

    "nature": {
        "desc": (
            "post about nature, environment, wildlife and outdoor activities"
        ),
        "keywords": ["nature", "forest", "animal", "wildlife", "environment"],
        "negative": []
    },

    "education": {
        "desc": (
            "post teaching something, tutorials and learning content"
        ),
        "keywords": ["learn", "tutorial", "guide", "education", "lesson"],
        "negative": []
    },

    "science": {
        "desc": (
            "post about science, research, physics, biology and discoveries"
        ),
        "keywords": ["science", "research", "experiment", "physics", "biology"],
        "negative": []
    },

    "history": {
        "desc": (
            "post about historical events, wars and past civilizations"
        ),
        "keywords": ["history", "war", "ancient", "empire", "civilization"],
        "negative": []
    },

    "entertainment": {
        "desc": (
            "post about movies, tv shows, anime and pop culture"
        ),
        "keywords": ["movie", "anime", "series", "tv", "entertainment"],
        "negative": []
    },

    "news": {
        "desc": (
            "post reporting news, updates and current events"
        ),
        "keywords": ["news", "update", "breaking", "report"],
        "negative": []
    },

    "politics": {
        "desc": (
            "post about politics, government and society"
        ),
        "keywords": ["politics", "government", "election", "law"],
        "negative": []
    },

    "promotional": {
        "desc": (
            "post promoting something, advertising or investment opportunity"
        ),
        "keywords": ["buy", "profit", "earn", "referral", "promo"],
        "negative": []
    },
}

CATEGORIES = list(CATEGORY_MAP.keys())

print("Carregando modelos...")

text_model = SentenceTransformer('all-MiniLM-L6-v2')
lang_model = fasttext.load_model("lid.176.bin")

cat_embeddings = text_model.encode(
    [CATEGORY_MAP[c]["desc"] for c in CATEGORIES],
    convert_to_tensor=True
)

print("Modelos carregados.")

# ==============================
# CLEAN TEXT
# ==============================

def clean_text(text):
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\w\sÀ-ÿ#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# ==============================
# LANGUAGE
# ==============================

def detect_language(text):
    pred = lang_model.predict(text[:500])
    return pred[0][0].replace("__label__", "")

# ==============================
# IMAGE
# ==============================

def extract_image(metadata, body):
    try:
        meta = json.loads(metadata)
        if "image" in meta and len(meta["image"]) > 0:
            return meta["image"][0]
    except:
        pass

    match = re.search(r'<img[^>]+src="([^">]+)"', body)
    if match:
        return match.group(1)

    return None

# ==============================
# KEYWORD SCORE (melhorado)
# ==============================

def keyword_score(text, keywords):
    score = 0
    for kw in keywords:
        occurrences = text.count(kw)
        if occurrences > 0:
            score += min(occurrences, 3)  # evita spam

    return score / max(len(keywords), 1)

# ==============================
# NEGATIVE SCORE
# ==============================

def negative_score(text, negatives):
    score = 0
    for kw in negatives:
        if kw in text:
            score += 1
    return score / max(len(negatives), 1)

# ==============================
# CLASSIFICAÇÃO HÍBRIDA
# ==============================

def classify_text(text):
    emb = text_model.encode(text, convert_to_tensor=True)
    sim_scores = util.cos_sim(emb, cat_embeddings)[0]

    results = []

    for i, cat in enumerate(CATEGORIES):
        sim = float(sim_scores[i])

        kw_score = keyword_score(text, CATEGORY_MAP[cat]["keywords"])
        neg_score = keyword_score(text, CATEGORY_MAP[cat]["negative"])

        # peso dinâmico
        kw_weight = 0.3 + (0.4 * kw_score)

        final_score = sim + (kw_score * kw_weight)

        # penalidade forte
        if neg_score > 0:
            final_score *= (1 - min(neg_score, 0.7))

        if final_score > 0.25:
            results.append({
                "name": cat,
                "confidence": round(final_score, 4)
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    # boost top1
    if results:
        results[0]["confidence"] = round(results[0]["confidence"] * 1.1, 4)

    return results[:3]

# ==============================
# MAIN
# ==============================

def main():
    with open(INPUT_FILE) as f:
        posts = json.load(f)

    print(f"{len(posts)} posts carregados.")

    results = []

    for i, post in enumerate(posts):
        text = clean_text(post["body"])

        if len(text.split()) < 20:
            continue

        if len(text) < 50:
            continue

        lang = detect_language(text)
        text_tags = classify_text(text)
        image_url = extract_image(post["json_metadata"], post["body"])

        results.append({
            "author": post["author"],
            "permlink": post["permlink"],
            "url": f"https://peakd.com/@{post['author']}/{post['permlink']}",
            "created": post["created"],
            "language": lang,
            "categories": text_tags,
            "image": image_url,
            "has_image": image_url is not None
        })

        if i % 50 == 0:
            print(f"Processados: {i}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("Finalizado.")

# ==============================

if __name__ == "__main__":
    main()
