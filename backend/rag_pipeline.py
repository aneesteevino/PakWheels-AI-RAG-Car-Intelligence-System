"""
RAG Pipeline for PakWheels Used Car Dataset
- Preprocessing & text-chunk generation
- Sentence-Transformer embeddings
- FAISS vector index
- Query classification & structured filtering
- Groq LLM integration
"""

import os
import re
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

import faiss
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using TF-IDF fallback")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the application.")
GROQ_MODEL   = "llama3-70b-8192"
EMBED_MODEL  = "all-MiniLM-L6-v2"          # fast, 384-dim, good quality
INDEX_CACHE  = Path(__file__).parent.parent / "data" / "faiss_index.pkl"

PKR_MILLION = 1_000_000


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_price(p: float) -> str:
    if pd.isna(p):
        return "N/A"
    m = p / PKR_MILLION
    return f"PKR {m:.2f}M" if m >= 1 else f"PKR {int(p):,}"


def fmt_mileage(m: float) -> str:
    if pd.isna(m):
        return "N/A"
    return f"{int(m):,} km"


def row_to_text(row: pd.Series) -> str:
    """Convert a DataFrame row into a natural-language text chunk for embedding."""
    parts = [
        f"{row.get('make','')} {row.get('model','')} {int(row['year']) if not pd.isna(row.get('year', float('nan'))) else ''}".strip(),
        f"City: {row.get('ad_city', 'Unknown')}",
        f"Body: {row.get('body', 'Unknown')}",
        f"Engine: {row.get('engine_cc', 'Unknown')}cc" if not pd.isna(row.get('engine_cc', float('nan'))) else "",
        f"Fuel: {row.get('fuel_type', 'Unknown')}",
        f"Transmission: {row.get('transmission', 'Unknown')}",
        f"Assembly: {row.get('assembly', 'Unknown')}",
        f"Color: {row.get('color', 'Unknown')}",
        f"Registered: {row.get('registered', 'Unknown')}",
        f"Mileage: {fmt_mileage(row.get('mileage', float('nan')))}",
        f"Price: {fmt_price(row.get('price', float('nan')))}",
    ]
    return ", ".join(p for p in parts if p)


def row_to_dict(row: pd.Series) -> dict:
    """Serialise a row for the API response (JSON-safe)."""
    return {
        "ad_ref":        str(row.get("ad_ref", "")),
        "make":          str(row.get("make", "")),
        "model":         str(row.get("model", "")),
        "year":          int(row["year"]) if not pd.isna(row.get("year", float("nan"))) else None,
        "city":          str(row.get("ad_city", "")),
        "body":          str(row.get("body", "")),
        "engine_cc":     int(row["engine_cc"]) if not pd.isna(row.get("engine_cc", float("nan"))) else None,
        "fuel_type":     str(row.get("fuel_type", "")),
        "transmission":  str(row.get("transmission", "")),
        "assembly":      str(row.get("assembly", "")),
        "color":         str(row.get("color", "")),
        "registered":    str(row.get("registered", "")),
        "mileage":       int(row["mileage"]) if not pd.isna(row.get("mileage", float("nan"))) else None,
        "price":         int(row["price"])   if not pd.isna(row.get("price",   float("nan"))) else None,
        "price_fmt":     fmt_price(row.get("price", float("nan"))),
        "text_snippet":  row_to_text(row),
    }


# ── Query Classifier ──────────────────────────────────────────────────────────
class QueryClassifier:
    """Rule-based query classifier to decide retrieval strategy."""

    PATTERNS = {
        "average":   r"\b(average|avg|mean)\b",
        "cheapest":  r"\b(cheap|cheapest|lowest price|minimum price|min price)\b",
        "expensive": r"\b(expensive|most expensive|highest price|max price)\b",
        "filter":    r"\b(under|below|above|over|between|more than|less than|after|before)\b",
        "compare":   r"\bvs\.?\b|\bversus\b|\bcompare\b",
        "list":      r"\b(show|list|find|give me|display)\b",
        "count":     r"\b(how many|count|number of)\b",
    }

    @classmethod
    def classify(cls, query: str) -> str:
        q = query.lower()
        for qtype, pattern in cls.PATTERNS.items():
            if re.search(pattern, q):
                return qtype
        return "general"


# ── Structured Filter Extractor ───────────────────────────────────────────────
class FilterExtractor:
    """Extract structured filters from a natural-language query."""

    CITIES = [
        "karachi", "lahore", "islamabad", "rawalpindi", "faisalabad",
        "multan", "peshawar", "quetta", "sialkot", "gujranwala",
        "hyderabad", "abbottabad", "bahawalpur", "sargodha",
    ]
    FUELS        = ["petrol", "diesel", "hybrid", "electric", "cng", "lpg"]
    TRANSMISSIONS= ["automatic", "manual"]
    ASSEMBLIES   = ["local", "imported"]
    BODY_TYPES   = ["suv", "sedan", "hatchback", "van", "pickup", "crossover",
                    "coupe", "wagon", "minivan", "truck"]

    @classmethod
    def extract(cls, query: str) -> dict:
        q = query.lower()
        filters: dict[str, Any] = {}

        # City
        for city in cls.CITIES:
            if city in q:
                filters["city"] = city.title()
                break

        # Fuel
        for fuel in cls.FUELS:
            if fuel in q:
                filters["fuel_type"] = fuel.capitalize()
                break

        # Transmission
        for t in cls.TRANSMISSIONS:
            if t in q:
                filters["transmission"] = t.capitalize()
                break

        # Assembly
        for a in cls.ASSEMBLIES:
            if a in q:
                filters["assembly"] = a.capitalize()
                break

        # Body type
        for b in cls.BODY_TYPES:
            if b in q:
                filters["body"] = b.upper() if b == "suv" else b.capitalize()
                break

        # Make / model (simple heuristic – capitalised words after common car brands)
        makes = ["toyota", "honda", "suzuki", "daihatsu", "kia", "hyundai",
                 "nissan", "mitsubishi", "mercedes", "bmw", "audi", "changan",
                 "proton", "mg", "haval", "prince", "united", "regal",
                 "peugeot", "jeep", "land rover", "ford", "volkswagen"]
        for make in makes:
            if make in q:
                filters["make"] = make.title()
                # Try to grab the model word that follows
                m = re.search(rf"{make}\s+(\w+)", q)
                if m:
                    next_word = m.group(1)
                    skip = {"in", "from", "with", "under", "over", "price", "automatic", "manual"}
                    if next_word not in skip:
                        filters["model_hint"] = next_word.capitalize()
                break

        # Price ceiling:  "under X million" / "below X lakh"
        m = re.search(r"under\s+(\d+(?:\.\d+)?)\s*(million|m\b|lakh|lac)", q)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            filters["price_max"] = int(val * (PKR_MILLION if "m" in unit or "million" in unit else 100_000))

        m = re.search(r"below\s+(\d+(?:\.\d+)?)\s*(million|m\b|lakh|lac)", q)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            filters["price_max"] = int(val * (PKR_MILLION if "m" in unit or "million" in unit else 100_000))

        # Raw numeric price (e.g. "under 3000000")
        m = re.search(r"under\s+(\d{5,})", q)
        if m and "price_max" not in filters:
            filters["price_max"] = int(m.group(1))

        # Year filters
        m = re.search(r"after\s+(20\d\d|19\d\d)", q)
        if m:
            filters["year_min"] = int(m.group(1))
        m = re.search(r"before\s+(20\d\d|19\d\d)", q)
        if m:
            filters["year_max"] = int(m.group(1))
        m = re.search(r"\b(20\d\d|19\d\d)\b", q)
        if m and "year_min" not in filters and "year_max" not in filters:
            filters["year_exact"] = int(m.group(1))

        # Mileage ceiling
        m = re.search(r"(?:low mileage|under|below)\s+(\d+)\s*(km|k\b)", q)
        if m:
            filters["mileage_max"] = int(m.group(1)) * (1000 if m.group(2).startswith("k") and len(m.group(2)) == 1 else 1)

        return filters


# ── RAG Pipeline ──────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.texts: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedder = SentenceTransformer(EMBED_MODEL)
            self.use_tfidf = False
        else:
            self.embedder = TfidfVectorizer(max_features=1000, stop_words='english')
            self.use_tfidf = True
            
        self.groq = Groq(api_key=GROQ_API_KEY)
        self.classifier  = QueryClassifier()
        self.filter_ext  = FilterExtractor()

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_data(self):
        log.info(f"Loading dataset from {self.data_path} …")
        df = pd.read_csv(self.data_path, low_memory=False)

        # Normalise columns
        df.columns = [c.strip().lower() for c in df.columns]
        if "ad_city" not in df.columns and "city" in df.columns:
            df = df.rename(columns={"city": "ad_city"})
        if "engine_cc" not in df.columns and "engine" in df.columns:
            df = df.rename(columns={"engine": "engine_cc"})

        # Type coercions
        df["price"]     = pd.to_numeric(df["price"],     errors="coerce")
        df["mileage"]   = pd.to_numeric(df["mileage"],   errors="coerce")
        df["year"]      = pd.to_numeric(df["year"],      errors="coerce")
        df["engine_cc"] = pd.to_numeric(df["engine_cc"], errors="coerce")

        # Drop rows with no price
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0]

        # Normalise string columns
        for col in ["make", "model", "ad_city", "body", "fuel_type",
                    "transmission", "assembly", "color", "registered"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                df[col] = df[col].replace("Nan", pd.NA)

        # Fix assembly column (often all-NaN in v02; fill from context)
        if df["assembly"].isna().all() and "assembly" in df.columns:
            df["assembly"] = "Unknown"

        log.info(f"Dataset loaded: {len(df):,} rows")
        self.df = df

    # ── Index building ────────────────────────────────────────────────────────
    def build_index(self, force: bool = False):
        self._load_data()

        if INDEX_CACHE.exists() and not force:
            log.info("Loading cached FAISS index …")
            with open(INDEX_CACHE, "rb") as f:
                payload = pickle.load(f)
            self.texts = payload["texts"]
            
            if self.use_tfidf:
                # For TF-IDF, we store the fitted vectorizer and matrix
                self.embedder = payload["vectorizer"]
                self.tfidf_matrix = payload["tfidf_matrix"]
                log.info(f"TF-IDF index loaded ({len(self.texts):,} documents).")
            else:
                # For sentence transformers, we use FAISS
                vecs = payload["vecs"]
                self.index = faiss.IndexFlatIP(vecs.shape[1])
                self.index.add(vecs)
                log.info(f"FAISS index loaded ({self.index.ntotal:,} vectors).")
            return

        log.info("Building text chunks …")
        self.texts = [row_to_text(row) for _, row in self.df.iterrows()]

        if self.use_tfidf:
            log.info("Generating TF-IDF vectors (fallback mode) …")
            self.tfidf_matrix = self.embedder.fit_transform(self.texts)
            log.info(f"TF-IDF index built: {len(self.texts):,} documents.")
            
            # Cache to disk
            INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_CACHE, "wb") as f:
                pickle.dump({
                    "texts": self.texts, 
                    "vectorizer": self.embedder,
                    "tfidf_matrix": self.tfidf_matrix
                }, f)
        else:
            log.info("Generating embeddings (this takes ~1-2 min for 77k rows) …")
            batch = 512
            all_vecs = []
            for i in range(0, len(self.texts), batch):
                chunk = self.texts[i: i + batch]
                vecs = self.embedder.encode(chunk, normalize_embeddings=True,
                                            show_progress_bar=False)
                all_vecs.append(vecs)
                if (i // batch) % 20 == 0:
                    log.info(f"  Embedded {min(i+batch, len(self.texts)):,}/{len(self.texts):,}")

            all_vecs = np.vstack(all_vecs).astype("float32")
            self.index = faiss.IndexFlatIP(all_vecs.shape[1])
            self.index.add(all_vecs)
            log.info(f"FAISS index built: {self.index.ntotal:,} vectors.")

            # Cache to disk
            INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_CACHE, "wb") as f:
                pickle.dump({"texts": self.texts, "vecs": all_vecs}, f)
                
        log.info("Index cached to disk.")

    # ── Apply structured filters ──────────────────────────────────────────────
    def _apply_filters(self, filters: dict) -> pd.DataFrame:
        df = self.df.copy()

        if "city" in filters:
            df = df[df["ad_city"].str.lower() == filters["city"].lower()]
        if "fuel_type" in filters:
            df = df[df["fuel_type"].str.lower() == filters["fuel_type"].lower()]
        if "transmission" in filters:
            df = df[df["transmission"].str.lower() == filters["transmission"].lower()]
        if "assembly" in filters:
            df = df[df["assembly"].str.lower() == filters["assembly"].lower()]
        if "body" in filters:
            df = df[df["body"].str.lower() == filters["body"].lower()]
        if "make" in filters:
            df = df[df["make"].str.lower() == filters["make"].lower()]
        if "model_hint" in filters:
            df = df[df["model"].str.lower().str.contains(filters["model_hint"].lower(), na=False)]
        if "price_max" in filters:
            df = df[df["price"] <= filters["price_max"]]
        if "year_min" in filters:
            df = df[df["year"] >= filters["year_min"]]
        if "year_max" in filters:
            df = df[df["year"] <= filters["year_max"]]
        if "year_exact" in filters:
            df = df[df["year"] == filters["year_exact"]]
        if "mileage_max" in filters:
            df = df[df["mileage"] <= filters["mileage_max"]]

        return df

    # ── Semantic retrieval ────────────────────────────────────────────────────
    def _semantic_retrieve(self, query: str, top_k: int, df_subset: pd.DataFrame) -> pd.DataFrame:
        """Embed the query, search FAISS or TF-IDF, filter to subset indices."""
        
        if self.use_tfidf:
            # TF-IDF approach
            query_vec = self.embedder.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top indices
            search_k = min(top_k * 20, len(similarities))
            top_indices = similarities.argsort()[-search_k:][::-1]
            
        else:
            # Sentence transformer + FAISS approach
            q_vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            search_k = min(top_k * 20, self.index.ntotal)
            _, top_indices = self.index.search(q_vec, search_k)
            top_indices = top_indices[0]

        # Map indices → DataFrame rows that are in the subset
        subset_positions = set(df_subset.index)
        df_idx_list = list(self.df.index)

        results = []
        for i in top_indices:
            if i < 0 or i >= len(df_idx_list):
                continue
            real_idx = df_idx_list[i]
            if real_idx in subset_positions:
                results.append(real_idx)
            if len(results) >= top_k:
                break

        # Fall back: if semantic finds too few results in subset, take price-sorted top rows
        if len(results) < top_k and len(df_subset) > 0:
            extra = df_subset.sort_values("price").index.tolist()
            for idx in extra:
                if idx not in results:
                    results.append(idx)
                if len(results) >= top_k:
                    break

        return self.df.loc[results] if results else df_subset.head(top_k)

    # ── Compute statistics ────────────────────────────────────────────────────
    def _compute_stats(self, df: pd.DataFrame, query_type: str) -> dict:
        if df.empty:
            return {}
        stats = {
            "count":    int(len(df)),
            "avg_price": int(df["price"].mean()) if not df["price"].isna().all() else None,
            "min_price": int(df["price"].min())  if not df["price"].isna().all() else None,
            "max_price": int(df["price"].max())  if not df["price"].isna().all() else None,
        }
        if stats["avg_price"]:
            stats["avg_price_fmt"] = fmt_price(stats["avg_price"])
            stats["min_price_fmt"] = fmt_price(stats["min_price"])
            stats["max_price_fmt"] = fmt_price(stats["max_price"])
        return stats

    # ── LLM call ──────────────────────────────────────────────────────────────
    def _call_llm(self, query: str, context_cars: list[dict], stats: dict,
                  query_type: str, filters: dict) -> str:
        context_lines = []
        for i, car in enumerate(context_cars[:5], 1):
            context_lines.append(f"{i}. {car['text_snippet']}")
        context_str = "\n".join(context_lines) if context_lines else "No specific cars found."

        stats_str = json.dumps(stats, indent=2) if stats else "N/A"
        filter_str = json.dumps(filters, indent=2) if filters else "No specific filters applied"

        system_prompt = """You are PakWheels AI, an expert assistant for Pakistan's used car market.
You have access to real listings data from PakWheels scraped dataset of ~77,000 used cars.
Answer questions accurately based ONLY on the provided data context and statistics.
- Be concise and helpful
- Always mention prices in PKR (use millions like "3.2M" for readability)
- If comparing cars, be objective
- If data is limited, say so honestly
- Format your response cleanly with bullet points or tables when listing cars
- Never make up prices or car details not in the context"""

        user_prompt = f"""User Query: "{query}"

Query Type: {query_type}
Filters Applied: {filter_str}

Computed Statistics from Dataset:
{stats_str}

Relevant Car Listings Retrieved:
{context_str}

Based on the above data, provide a clear, accurate answer to the user's question."""

        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    # ── Main query entrypoint ─────────────────────────────────────────────────
    def query(self, user_query: str, top_k: int = 5) -> dict:
        query_type = self.classifier.classify(user_query)
        filters    = self.filter_ext.extract(user_query)

        log.info(f"Query: '{user_query}' | type={query_type} | filters={filters}")

        # 1. Apply structured filters
        filtered_df = self._apply_filters(filters)
        log.info(f"After filters: {len(filtered_df):,} rows")

        # 2. Compute stats on filtered subset
        stats = self._compute_stats(filtered_df, query_type)

        # 3. Semantic retrieval on filtered subset
        if len(filtered_df) == 0:
            log.warning("No rows after filtering — falling back to full dataset")
            filtered_df = self.df
            stats = {}

        retrieved_df = self._semantic_retrieve(user_query, top_k, filtered_df)

        # 4. Serialise cars
        retrieved_cars = [row_to_dict(row) for _, row in retrieved_df.iterrows()]

        # 5. LLM answer
        answer = self._call_llm(user_query, retrieved_cars, stats, query_type, filters)

        return {
            "answer":        answer,
            "retrieved_cars": retrieved_cars,
            "stats":         stats,
            "query_type":    query_type,
            "filters_applied": filters,
        }
