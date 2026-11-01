"""Knowledge base: load kb/*.md, chunk by [SECTION-ID], retrieve with BM25.

Design notes
- One chunk per "## [ID] Title" section; header metadata (doc_type, status,
  effective_from) is kept with every chunk.
- Superseded documents are excluded (RET-3 already covers the old electronics rule).
- Scores are weighted by document precedence: policy > agent guide > help > FAQ > marketing.
- A few "companion" sections are added when their partner is retrieved, because
  some answers need two sources (e.g. refund timeline + business-day holidays).
- Product rows from data/products.csv are found by SKU or product name.
"""
import csv
import glob
import os
import re

from rank_bm25 import BM25Okapi

ROOT = os.getenv("DATASET_DIR", "data_local/Capstone_Project_Dataset_FTTT_2026")

SEC_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*?)\s*$", re.M)
TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097f\u20b9]+")
STOP = set(
    "a an the is are was were be to of in on for and or if it its my me i you your we our "
    "do does did can could will would should how what when where which who this that these "
    "those with at by from as than then so but any have has had am".split()
)

TYPE_WEIGHT = {"policy": 1.0, "agent_guide": 0.95, "meta": 0.9,
               "help": 0.85, "faq": 0.7, "marketing": 0.5}

# customer wording -> document wording (kept small; the LLM rewrite handles the rest)
SYNONYMS = [
    (r"send (it )?back|give (it )?back|take (it )?back|\bwapas\b|\bvapas\b|वापस|लौटा", "return reverse pickup"),
    (r"nobody (came|collected)|no ?one (came|collected)|didn.t come|not collected|\bmissed\b", "missed pickup reverse pickup"),
    (r"money back|\bpaise\b|\bpaisa\b|पैसे|रिफ़ंड|रिफंड", "refund"),
    (r"कैंसिल|रद्द|\bcancel", "cancellation cancel"),
    (r"broken|cracked|\btorn\b|defect|faulty|not working|damag|खराब|टूट", "damaged defective"),
    (r"not (yet )?(arrived|received|delivered)|hasn.t arrived|still waiting|\blate\b|delay|देर", "late delivery promised date"),
    (r"didn.t (order|make)|fraud|scam|unauthori[sz]ed|hack|police|court|lawyer|legal", "unauthorised legal escalate human"),
    (r"\bhuman\b|\bagent\b|\bperson\b|manager|colleague", "human colleague handover"),
    (r"swell|burning|smoke|spark|\bfire\b|shock|overheat", "safety swelling battery burning"),
    (r"guarantee|warranty|वारंटी|assured", "warranty assured"),
    (r"\bcod\b|cash on delivery", "cash on delivery cod"),
    (r"grocery|\bslot\b", "grocery delivery slot"),
    (r"वॉलेट|wallet", "wallet"),
    (r"\bplus\b", "plus membership"),
    (r"coupon|promo|voucher|discount", "coupon"),
    (r"opened|\bopen\b|खोल|\bused\b|tried|seal", "unopened seal intact condition of returned items"),
    (r"lipstick|perfume|parfum|cream|shampoo|serum|makeup|beauty|लिपस्टिक|ब्यूटी", "beauty personal care"),
    (r"कितने दिन|how many days|kitne din|return window|how long.*return", "return window days category"),
    (r"shoe|sneaker|jeans|shirt|kurta|dress|jacket|कपड़े|जूते", "clothing footwear fashion"),
    (r"exchange|size|colour|color", "exchange size colour"),
]

# companion sections: retrieving the key also pulls the values
LINKS = {
    "REF-2": ["KB-00.3", "REF-1", "KB-00.1"],
    "REF-1": ["REF-2"],
    "RET-2": ["RET-3", "RET-8"],
    "RET-7": ["REF-2"],
    "RET-6": ["PLS-2", "DWM-2"],
    "DEL-4": ["DEL-2", "KB-00.1"],
    "PLS-2": ["RET-6", "RET-7", "KB-00.1"],
    "DWM-3": ["ESC-1", "ESC-2"],
    "ESC-1": ["ESC-2", "ESC-3"],
    "CAN-2": ["REF-1", "REF-2"],
    "TRK-2": ["DWM-1"],
    "WAR-2": ["MKS-3"],
    "CAN-1": ["REF-1"],
}

NUMW = {"zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2",
        "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9"}


def _stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def tokenize(text):
    return [_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def expand(query):
    q = query.lower()
    extra = [add for pat, add in SYNONYMS if re.search(pat, q)]
    return query + " " + " ".join(extra)


def _header(text):
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, (text[m.end():] if m else text)


class KB:
    def __init__(self, include_superseded=False):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(ROOT, "kb", "*.md"))):
            with open(path, encoding="utf-8") as f:
                meta, body = _header(f.read())
            if meta.get("status") == "superseded" and not include_superseded:
                continue
            heads = list(SEC_RE.finditer(body))
            for i, h in enumerate(heads):
                end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
                text = body[h.end():end].strip()
                self.chunks.append({
                    "id": h.group(1).strip(),
                    "sec_title": h.group(2),
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("title", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "status": meta.get("status", ""),
                    "effective_from": meta.get("effective_from", ""),
                    "language": meta.get("language", "en"),
                    "body": text,
                })
        self.by_id = {c["id"]: c for c in self.chunks}
        corpus = [tokenize(f'{c["doc_title"]} {c["sec_title"]} {c["body"]}') for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        self.weight = [TYPE_WEIGHT.get(c["doc_type"], 0.8) for c in self.chunks]
        self.products = self._load_products()
        pre = {p["sku"].split("-")[0] for p in self.products}
        self.prefixes = sorted(pre | {"ELE", "HOM", "FSH", "BKS", "BTY", "GRO"})

    @staticmethod
    def _load_products():
        path = os.path.join(ROOT, "data", "products.csv")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    # ---- speech-text clean-up: "ele zero zero one to" -> "ELE-0012"
    def normalize_speech(self, text):
        words = "|".join(NUMW)
        pat = re.compile(r"\b(" + "|".join(self.prefixes) + r")[\s-]+((?:(?:" + words + r")\s*){4})", re.I)

        def fix(m):
            digits = "".join(NUMW[w.lower()] for w in m.group(2).split())
            return f"{m.group(1).upper()}-{digits} "
        return pat.sub(fix, text)

    def search(self, query, k=5, with_links=True, max_extra=4):
        q = self.normalize_speech(query)
        toks = tokenize(expand(q))
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = sorted(range(len(self.chunks)),
                       key=lambda i: scores[i] * self.weight[i], reverse=True)
        hits = []
        for i in order[:k]:
            if scores[i] > 0:
                hits.append({**self.chunks[i], "score": float(scores[i] * self.weight[i])})
        if with_links:
            have = {h["id"] for h in hits}
            extra = 0
            for h in list(hits):
                for lid in LINKS.get(h["id"], []):
                    if lid in self.by_id and lid not in have and extra < max_extra:
                        hits.append({**self.by_id[lid], "score": 0.0})
                        have.add(lid)
                        extra += 1
        return hits

    def products_for(self, text, limit=4):
        t = self.normalize_speech(text)
        skus = set(re.findall(r"\b([A-Z]{3}-\d{4})\b", t.upper()))
        rows = [p for p in self.products if p["sku"] in skus]
        if not rows:
            low = t.lower()
            for p in self.products:
                full = p["title"].lower()
                base = p["title"].split(", ")[0].lower()
                if full in low:
                    rows = [p] + [r for r in rows if r is not p]
                    break
                if len(base) > 6 and base in low:
                    rows.append(p)
        return rows[:limit]


def format_context(chunks, products=()):
    parts = []
    for c in chunks:
        parts.append(
            f'[{c["id"]}] {c["doc_title"]} | {c["doc_type"]}, {c["status"]}, '
            f'effective {c["effective_from"]}\n{c["body"]}'
        )
    for p in products:
        parts.append(
            f'[products.csv:{p["sku"]}] {p["title"]} | return_category={p["return_category"]}, '
            f'price=Rs {p["price_inr"]}, fulfilled_by={p["fulfilled_by"]}, '
            f'assured={p["assured"]}, warranty_months={p["warranty_months"]}'
        )
    return "\n\n".join(parts)