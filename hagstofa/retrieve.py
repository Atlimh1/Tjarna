"""Find candidate tables for a question. Several interchangeable retrievers, all with
search(question, k) -> [(table, score), ...], so eval/run_eval.py can compare them.

  raw     character n-grams over raw text incl. English and value labels (the first baseline)
  lemma   BÍN base forms (Miðeind's islenska) of titles, variables, English and value labels, char n-grams,
          plus SYNONYMS query expansion; n-grams over lemmas also match compounds (þorskafli ~ afli)
  embed   bge-m3 dense embeddings of titles/variables: matches meaning ("verðbólga" ~ "vísitala neysluverðs")
  fusion  raw + lemma + embed merged by reciprocal rank fusion (DEFAULT)

Held-out dev set (270 generated questions), top1 / top3 / top8:
  raw 31/53/72   lemma 35/53/72   embed 30/50/65   fusion 44/66/81   (fusion before Places: 42/66/82)
The three kinds catch different questions (exact values / inflection + synonyms / meaning), which is
why merging them gains ~10 points. lemma and embed down-weight archive tables (eldra efni, Sögulegar, census).
search() first fixes spelling (correct_question, not for English); raw folds accents on both sides.
Together they lift dev without accents from 33/58/73 to 40/63/77 (eval/experiments/greynir_correct.py).
"""
import difflib
import functools
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import tokenizer
from islenska import Bin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA = Path(__file__).resolve().parent.parent / "data"
CATALOG = DATA / "catalog.jsonl"
QUESTION_VECTORS = DATA / "question_vectors.sqlite"
CORRECTIONS = DATA / "corrections.sqlite"
ARCHIVE_WEIGHT = 0.6
MAX_VALUES_INDEXED = 40  # per variable
# Question names a town -> scores of tables that contain it are multiplied by 1 + boost (see Places)
PLACE_BOOST = {"lemma": 0.5, "fusion": 0.2}
# The place boost only for tables scoring at least this share of the best (on the question's topic): "bílar í
# Reykjavík" lifted every Reykjavík table above the national vehicles table. 0 = every table with the place (as before).
# Measured 1 Nov 13:00 (search only, four-way, top 1/3/12): 0.5 dev 154/213/253 (vs 252), localities 5/11/12 (vs
# 6/12/12), set 1 28/39/51 (vs 28/41/51); 0.7 also loses muni, real and set 1 at 12. Kept at 0 (off).
PLACE_TOPIC_MIN = float(os.environ.get("PLACE_TOPIC_MIN", "0"))
# Lemmas of municipality values that are ordinary words ("Sveitarfélagið Ölfus", "Rangárþing eystra", "Alls",
# "Ótilgreint lögheimili", "Utan sveitarfélaga"); they'd fire the boost on "Norðurlandi eystra" or "til útlanda"
PLACE_GENERIC = set(
    "sveitarfélag alls allur eystri eystra vestri vestra ytri ytra fremri utan útland ótilgreindur "
    "lögheimili lögheimila lögheimill garður enni".split()
)
# A fine-tuned model is used by setting EMBED_MODEL=models/<name> (scripts/finetune_embed.py); its table vectors
# get their own cache file, computed on first use (minutes on a GPU, ~1.5 h on a CPU)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
# A second embedding model in the same fusion (our fine-tuned bge-m3, scripts/finetune_jureca.sbatch): set
# EMBED_MODEL_2=<its folder>. Four-way fusion, 31 Oct (eval/experiments/candidate_union.py), right first/top 3/top 8:
# dev 51/72/87 -> 57/78/91, English dev 46/70/83 -> 54/78/87, hand 56/78/92 -> 61/81/97. Unset: three-way.
EMBED_MODEL_2 = os.environ.get("EMBED_MODEL_2")
STOPWORDS = set(
    "hve hvað hver hversu margur vera í á eiga ær um til frá með og eftir sem hafa það þessi sá ég þú hann "
    "hún að af við fyrir hjá yfir undir núna nú síðasta síðastur fyrra fyrri þegar þá "
    "the of in how many what is was were there a an".split()
)

# Query expansion: everyday word (BÍN base form, or English word) -> Hagstofa's official terms,
# appended to the question before search. Only adds words, so a wrong entry adds noise but can't hide a table.
SYNONYMS = {
    # population
    "búa": "mannfjöldi", "íbúi": "mannfjöldi", "íbúafjöldi": "mannfjöldi",
    "population": "mannfjöldi", "live": "mannfjöldi", "people": "mannfjöldi", "inhabitants": "mannfjöldi",
    # births, deaths, migration, marriage
    "fæða": "fæddir", "born": "fæddir", "births": "fæddir",
    "láta": "dánir",  # "létust" is the middle voice of láta
    "died": "dánir", "deaths": "dánir",
    "flytja": "búferlaflutningar", "moved": "búferlaflutningar", "migration": "búferlaflutningar",
    "gifta": "hjónavígslur", "giftast": "hjónavígslur", "marriages": "hjónavígslur",
    # prices, labour, economy
    "verðbólga": "vísitala neysluverðs", "inflation": "vísitala neysluverðs", "verðlag": "vísitala neysluverðs",
    "atvinnuleysi": "atvinnulausir vinnuafl", "unemployment": "atvinnulausir vinnuafl atvinnuleysi",
    "hagvöxtur": "landsframleiðsla", "gdp": "landsframleiðsla", "growth": "landsframleiðsla",
    "wages": "laun", "salary": "laun",
    # tourism, transport, schools, fishing, trade, environment
    "ferðamaður": "farþegar gistinætur", "tourists": "farþegar gistinætur", "hotel": "gistinætur hótel",
    "bíll": "ökutæki", "cars": "ökutæki",
    "framhaldsskóli": "nemendur ofan grunnskóla", "menntaskóli": "nemendur ofan grunnskóla",
    "landa": "afli", "veiða": "afli",
    "exports": "útflutningur", "imports": "innflutningur",
    "emissions": "losun gróðurhúsalofttegunda",
    # elections, names, livestock (31 Oct: tables say kosningaþátttaka, eiginnöfn, búpeningur; people don't)
    "kjörsókn": "kosningaþátttaka", "turnout": "kosningaþátttaka kjörsókn",
    "heita": "eiginnöfn nöfn", "nafn": "eiginnöfn", "named": "eiginnöfn nöfn", "names": "eiginnöfn nöfn",
    "sauðfé": "búpeningur", "nautgripur": "búpeningur", "sheep": "búpeningur sauðfé",
    "cattle": "búpeningur nautgripir", "livestock": "búpeningur", "horses": "búpeningur hross",
    # immigrants are a value of "bakgrunnur" (MAN43000 Mannfjöldi eftir bakgrunni), not a word in its title
    "innflytjandi": "mannfjöldi eftir bakgrunni", "immigrants": "mannfjöldi eftir bakgrunni innflytjendur",
}

_bin = Bin()
log = logging.getLogger(__name__)


def load_catalog(path=CATALOG):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


@functools.lru_cache(maxsize=None)
def _lemmas_of(word):
    _, meanings = _bin.lookup(word)
    return tuple(sorted({m.ord.lower() for m in meanings})) or (word.lower(),)


def lemmas(text):
    """Icelandic text -> base forms of its words (all candidates for ambiguous words), stopwords removed."""
    out = []
    for tok in tokenizer.tokenize(text):
        if tok.kind == tokenizer.TOK.WORD:
            out.extend(lemma for lemma in _lemmas_of(tok.txt) if lemma not in STOPWORDS)
    return out


# "flytja inn / út" is trade, not migration: "hvað er flutt inn mikið af kaffi" found only migration tables
TRADE = re.compile(r"\b(flutt\w*|flyt\w*)\s+(inn|út|ut)\b", re.I)


def expand_query(question):
    """The question plus the official terms for any everyday words in it (see SYNONYMS)."""
    words = {w.lower() for w in question.split()} | set(lemmas(question))
    extra = list(dict.fromkeys(SYNONYMS[w] for w in sorted(words) if w in SYNONYMS))  # dedupe, keep order
    trade = TRADE.search(question)
    if trade:
        extra = [e for e in extra if e != SYNONYMS["flytja"]]
        extra.append("innflutningur vörur" if trade.group(2).lower() == "inn" else "útflutningur vörur")
    return " ".join([question, *extra]) if extra else question


FOLD = str.maketrans({"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ý": "y", "ö": "o", "ð": "d",
                      "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ý": "Y", "Ö": "O", "Ð": "D"})
ENGLISH = set("the of in how many what is was were are there did does do and to for by which who much has have per "
              "average".split())


def fold(text):
    """Text without Icelandic letters, as typed on a foreign keyboard: "háskólanáms" -> "haskolanams"."""
    return text.translate(FOLD).replace("þ", "th").replace("Þ", "Th").replace("æ", "ae").replace("Æ", "Ae")


def is_english(question):
    """Two English function words and no Icelandic letters outside capitalised words, so place names don't count
    ("How many babies were born in Kópavogur?" is English). Right on all 810 dev and misspelled-dev questions."""
    words = re.findall(r"\w+", question)
    return (sum(w.lower() in ENGLISH for w in words) >= 2
            and all(fold(w) == w for w in words if not w[:1].isupper()))


# Accent restoration without a model (31 Oct, eval/experiments/accents.py: no accents @1 43 -> 50, dev unchanged).
# ON by default (ACCENTS=0 turns it off), only for questions with no Icelandic letter:
#   1 = a word BÍN doesn't know gets the one common word that folds to it ("hvad" -> "hvað", "bua" -> "búa")
#   2 = also a known but rare word whose accented twin is far more common ("hatt" -> "hátt", "ar" -> "ár")
ACCENTS = os.environ.get("ACCENTS", "1")
ACCENT_CHOICES = {"a": "aá", "e": "eé", "i": "ií", "o": "oóö", "u": "uú", "y": "yý", "d": "dð"}
ACCENT_PAIRS = {"th": "þ", "ae": "æ"}  # how a foreign keyboard writes þ and æ (fold() does the same)
COMMON = 100  # Icegrams count (a 1-billion-word corpus) for a candidate to count as a common word
DOMINANT = 50  # mode 2: the accented twin must be this many times as common as the typed word
MAX_VARIANTS = 4000  # words with more accent combinations than this are left as typed


@functools.cache
def _exact_bin():
    return Bin(add_compounds=False)  # real BÍN entries only: the default also guesses compounds ("kopa-vogi")


@functools.cache
def _ngrams():
    from icegrams import Ngrams  # installed with GreynirCorrect

    return Ngrams()


def _in_bin(word):
    """A real BÍN entry: bin_id 0 marks a form islenska guessed ("öhöflega" from the -legur pattern)."""
    return any(m.bin_id for m in _exact_bin().lookup(word)[1])


@functools.cache
def _icelandic_catalog_words():
    """Words of the Icelandic catalog text only (English titles hold "the", "is", ...); a typed word found here is
    left as it is ("rvk", "vnv")."""
    words = set()
    for t in load_catalog():
        texts = [t["title_is"], *t["breadcrumbs_is"]]
        for v in t["variables_is"]:
            texts += [v["text"], *v["valueTexts"][:200]]
        words |= {w.lower() for w in re.findall(r"[^\W\d_]{3,}", " ".join(texts))}
    return frozenset(words)  # 3+ letters: "a" and "i" occur only as pieces of abbreviations ("ót.a.")


def _variants(word):
    """Every spelling that fold() turns into `word` ("bua" -> bua, búa, buá, búá), lowercase, at most MAX_VARIANTS."""
    out = [""]
    i = 0
    while i < len(word):
        pair = ACCENT_PAIRS.get(word[i:i + 2])
        if pair:  # "th" stays "th" or becomes "þ"
            rest = word[i + 2:]
            return list({v + pair + r for v in out for r in _variants(rest)}
                        | {v + word[i:i + 2] + r for v in out for r in _variants(rest)})[:MAX_VARIANTS]
        out = [v + c for v in out for c in ACCENT_CHOICES.get(word[i], word[i])]
        if len(out) > MAX_VARIANTS:
            return [word]
        i += 1
    return out


def _restore_word(word, mode):
    """The accented word, or the word as typed when there's no single clear candidate."""
    lower = word.lower()
    if fold(lower) != lower or not lower.isalpha() or lower in _icelandic_catalog_words():
        return word  # already has Icelandic letters, has digits, or the catalog uses it as typed ("GDP")
    known = _in_bin(lower) or _in_bin(word)
    if known and mode < 2:
        return word
    freq = _ngrams().freq
    cased = (lambda v: v[0].upper() + v[1:]) if word[0].isupper() else (lambda v: v)
    candidates = {cased(v): max(freq(v), freq(cased(v))) for v in _variants(lower)
                  if v != lower and (_in_bin(v) or _in_bin(cased(v)) or v in _icelandic_catalog_words())}
    common = sorted((n, v) for v, n in candidates.items() if n >= COMMON)
    if not common:
        return word
    if known:  # mode 2: replace a real word only when its accented twin dominates it
        best_n, best = common[-1]
        return best if best_n >= DOMINANT * max(freq(lower), 1) and (len(common) == 1 or best_n >= DOMINANT * common[-2][0]) else word
    if len(common) == 1:
        return common[0][1]
    return common[-1][1] if mode >= 2 and common[-1][0] >= DOMINANT * common[-2][0] else word


def restore_accents(question, mode=None):
    """Icelandic letters put back in a question typed entirely without them, word by word, without a model:
    "hvad bua margir i Kopavogi" -> "hvað búa margir í Kópavogi". English questions are left alone."""
    mode = int(ACCENTS if mode is None else mode)
    if not mode or is_english(question) or fold(question) != question:
        return question  # a question with any Icelandic letter wasn't typed on a foreign keyboard: there an
        # unaccented word is a typo ("skatar" for "skattar"), not a lost accent ("skátar"); measured in accents.py
    return re.sub(r"[^\W\d_]+", lambda m: _restore_word(m.group(0), mode), question)


_correct_lock = threading.Lock()


@functools.cache
def catalog_words():
    """Every word in the catalog's titles, breadcrumbs, variables and labels (IS and EN), lowercased: terms BÍN may
    not know ("vísitölu", "GDP") that a correction must leave alone."""
    words = set()
    for t in load_catalog():
        texts = [t["title_is"], t.get("title_en") or "", *t["breadcrumbs_is"]]
        for v in t["variables_is"] + (t.get("variables_en") or []):
            texts += [v["text"], *v["valueTexts"][:200]]
        words |= {w.lower() for w in re.findall(r"[^\W\d_]+", " ".join(texts))}
    return frozenset(words)


def _known(word):
    return bool(_bin.lookup(word)[1] or _bin.lookup(word.lower())[1]) or word.lower() in catalog_words()


def _accept(old, new):
    """Take GreynirCorrect's version of one word only where the typed word is not a word at all."""
    if fold(old.lower()) == fold(new.lower()):
        return True  # only capitals or Icelandic letters restored: "kopavogi" -> "Kópavogi", "hatt" -> "hátt"
    core = re.sub(r"^\W+|\W+$", "", old)
    if not core or any(c.isdigit() for c in core):
        return False  # numbers and ranges stay as typed ("2002-2003", "0-5")
    return not _known(core)


def safe_correction(question, fixed):
    """GreynirCorrect's fix, kept only where it repairs a word that isn't one (31 Oct: on 402 correctly written
    questions it changed 43, among them "sent" -> "send", "ennþá" -> "enn þá", and it dropped the second
    question of "Hvað búa margir á Íslandi? En á höfuðborgarsvæðinu?"). Words are never dropped or added."""
    a, b = question.split(), fixed.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            out += [new if _accept(old, new) else old for old, new in zip(a[i1:i2], b[j1:j2])]
        elif op != "insert":  # equal, or words GreynirCorrect dropped, merged or split: as typed
            out += a[i1:i2]
    return " ".join(out)


def correct_question(question):
    """GreynirCorrect's fix of an Icelandic question ("þjóðkyrkjunni" -> "þjóðkirkjunni", "Kopavogi" -> "Kópavogi");
    English is left alone, since GreynirCorrect turns English words into Icelandic ones. Takes ~0.1-1 s,
    so results are kept in data/corrections.sqlite (shared by the lab, eval runs and the app)."""
    if is_english(question):
        return question
    question = restore_accents(question)  # only with ACCENTS=1 or 2; before GreynirCorrect, which then sees real words
    with sqlite3.connect(CORRECTIONS, timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS fix (q TEXT PRIMARY KEY, fixed TEXT)")
        row = db.execute("SELECT fixed FROM fix WHERE q=?", (question,)).fetchone()
        if row:  # the cache keeps GreynirCorrect's own output; the filter runs on every read
            return safe_correction(question, row[0])
        from .icelandic import correct  # loads GreynirCorrect; icelandic imports this module

        with _correct_lock:  # GreynirCorrect isn't known to be thread-safe; the server runs questions in threads
            fixed = correct(question)[0]
        db.execute("INSERT OR REPLACE INTO fix VALUES (?, ?)", (question, fixed))
        return safe_correction(question, fixed)


def table_text(t):
    """Title (twice), breadcrumbs and variable names: what lemma and embed index."""
    variables = " ".join(v["text"] for v in t["variables_is"])
    return " ".join([t["title_is"], t["title_is"], " ".join(t["breadcrumbs_is"]), variables])


def is_municipality(v):
    return "sveitarf" in v["text"].lower()


def is_locality(v):
    """Towns and villages by their own names (byggðakjarnar, þéttbýlisstaðir), not by municipality."""
    return any(w in v["text"].lower() for w in ("byggðakjarn", "þéttbýli"))


TOTAL_PLACES = {"alls", "samtals", "dreifbýli", "annað dreifbýli", "þéttbýli", "stór-reykjavík"}


class Places:
    """Municipality names in the catalog and the tables that contain each, so a town named in the question
    can boost tables broken down by municipality. As text, 63 town names only dilute a table's topic words
    (TF-IDF vectors are length-normalized), so they're a separate signal instead.

    Names are values that appear with an official 4-digit code somewhere; that excludes party lists in
    election tables ("Kópavogur B Framsóknarflokkur"). Tables share the codes, so they also give aliases:
    1000 = Kópavogsbær / Kópavogur, so "í Kópavogi" finds tables that use the official name. Codes were
    reused after mergers (7502 = Fljótsdalshreppur and Vopnafjarðarhreppur), so only names sharing their
    first 4 letters count as aliases."""

    def __init__(self, tables):
        by_code = defaultdict(set)
        for t in tables:
            for v in filter(is_municipality, t["variables_is"]):
                for code, text in zip(v["values"], v["valueTexts"]):
                    text = text.strip()
                    if re.fullmatch(r"\d{4}", code) and re.fullmatch(r"[^\W\d][^\d()]*", text) and " - " not in text:
                        by_code[code].add(text)
        self.aliases = {
            n: [a for a in sorted(group) if a != n and a[:4].lower() == n[:4].lower()]
            for group in by_code.values()
            for n in group
        }
        years = [int(t["updated"][:4]) for t in tables if (t.get("updated") or "")[:4].isdigit()]
        since = str(max(years) - 1) if years else ""  # no dates known: every municipality counts as current
        current = {text.strip() for t in tables if (t.get("updated") or "")[:4] >= since
                   for v in filter(is_municipality, t["variables_is"])
                   for code, text in zip(v["values"], v["valueTexts"]) if re.fullmatch(r"\d{4}", code)}
        localities = set()
        for t in tables:  # localities too (Vík, Hvolsvöllur, Siglufjörður): their codes are short, not municipal
            for v in filter(is_locality, t["variables_is"]):
                for text in (x.strip() for x in v["valueTexts"]):
                    if re.fullmatch(r"[^\W\d][^\d(),]*", text) and text.lower() not in TOTAL_PLACES:
                        self.aliases.setdefault(text, [])
                        localities.add(text)
        self.names = set(self.aliases)
        # towns and villages that are no current municipality's name (Vík: Mýrdalshreppur; Selfoss: Árborg; the old
        # Selfosskaupstaður doesn't count), by accent-free lemma; Akureyri / Akureyrarbær share 4 letters: not one
        stems = {fold(m[:4].lower()) for m in current}
        self.village_lemmas = {fold(lemma): n for n in localities if fold(n[:4].lower()) not in stems
                               for lemma in lemmas(n.split(" í ")[0]) if lemma not in PLACE_GENERIC}
        in_title = re.compile(r"\b(" + "|".join(map(re.escape, sorted(self.names, key=len, reverse=True))) + r")\b")
        tables_with = defaultdict(set)
        for i, t in enumerate(tables):
            found = set(in_title.findall(t["title_is"]))  # "Úrslit borgarstjórnakosninga í Reykjavík 1930-2022"
            for v in t["variables_is"]:  # any variable: also weather stations, regions, constituencies
                for text in v["valueTexts"]:  # the name itself, or a list/era of it ("Reykjavík B-listi …")
                    words = text.split()
                    found |= {n for k in range(1, 5) if (n := " ".join(words[:k])) in self.names}
            for name in found:
                for lemma in lemmas(" ".join([name, *self.aliases[name]])):
                    if lemma not in PLACE_GENERIC:
                        tables_with[lemma].add(i)
        self.tables_with = {lemma: np.array(sorted(ids)) for lemma, ids in tables_with.items()}

    def villages(self, question):
        """Towns and villages named in the question that are not municipalities ("How many live in Vik?"). Only
        capitalised words count, so "búðir" (shops) is never the village Búðir."""
        names = " ".join(w for w in re.findall(r"\w+", correct_question(question)) if w[0].isupper())
        return sorted({self.village_lemmas[fold(lemma)] for lemma in lemmas(names) if fold(lemma) in self.village_lemmas})

    def village_tables(self, names, tables, most=2):
        """The newest tables with a towns/villages variable that lists these names: the candidates a village
        question needs when search ranked only municipality tables ("Vik": MAN030101 was not in the top 8)."""
        hits = [t for t in tables if any(is_locality(v) and any(x.strip() in names for x in v["valueTexts"])
                                         for v in t["variables_is"])]
        return sorted(hits, key=lambda t: t.get("updated") or "", reverse=True)[:most]

    def boost(self, question, scores, b):
        """scores with tables that contain a town named in the question multiplied by 1 + b."""
        hits = [self.tables_with[lemma] for lemma in set(lemmas(question)) if lemma in self.tables_with]
        if not hits:
            return scores
        scores = scores.copy()
        idx = np.unique(np.concatenate(hits))
        if PLACE_TOPIC_MIN > 0 and scores.max() > 0:  # only tables already close to the best on topic
            idx = idx[scores[idx] >= PLACE_TOPIC_MIN * scores.max()]
        scores[idx] *= 1 + b
        return scores


def lemma_table_text(t, places=frozenset()):
    """What lemma indexes: titles/variables, plus English and value labels. Dropping English and values
    looked fine on the 36 hand-written questions but cost 16 points top-1 on the held-out dev set.
    Town names in municipality variables are left out (Places handles them)."""
    english = [t.get("title_en") or ""] + [v["text"] for v in (t.get("variables_en") or [])]
    values = []
    for v in t["variables_is"]:
        texts = [x for x in v["valueTexts"] if x.strip() not in places] if is_municipality(v) else v["valueTexts"]
        values += texts[:MAX_VALUES_INDEXED]
    return " ".join([table_text(t), *english, *values])


def raw_table_text(t):
    """Everything incl. English and value labels: what the raw baseline indexes."""
    parts = [t["title_is"], t["title_is"], " ".join(t["breadcrumbs_is"]), t.get("title_en") or ""]
    for v in t["variables_is"] + (t.get("variables_en") or []):
        parts.append(v["text"])
        parts.extend(v["valueTexts"][:MAX_VALUES_INDEXED])
    return " ".join(parts).lower()


def is_archive(t):
    """Old series, censuses and the 2024 method-revision tables (4_endurskodun, MAN90001-4): special-purpose, not
    the current figures. "Hvað búa margir á Íslandi?" ranked the revision tables first (31 Oct dry run); as archives:
    dev 138 -> 139 right first, hand top 8 32 -> 33, nothing lost. (Raising the overview folders, 1_yfirlit, by 1.2
    instead cost dev 8 first places: not kept.)"""
    path = t["path"].lower()
    return "eldra" in path or "endurskodun" in path or t["path"].startswith("Sogulegar") or t["id"].startswith("CEN")


class _Base:
    archive_weight = ARCHIVE_WEIGHT
    expand = True  # apply SYNONYMS before searching
    correct = True  # fix spelling before searching (correct_question)

    def __init__(self, tables=None):
        self.tables = tables if tables is not None else load_catalog()
        self.weights = np.array([self.archive_weight if is_archive(t) else 1.0 for t in self.tables])

    def scores(self, question):
        raise NotImplementedError

    def search(self, question, k=8, correct=None):
        """correct=False when the caller already corrected."""
        if self.correct if correct is None else correct:
            question = correct_question(question)
        if self.expand:
            question = expand_query(question)
        s = self.scores(question) * self.weights
        top = s.argsort()[::-1][:k]
        return [(self.tables[i], float(s[i])) for i in top]


class RawRetriever(_Base):
    archive_weight = 1.0
    expand = correct = False  # the first baseline, apart from accent folding

    def __init__(self, tables=None):
        super().__init__(tables)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(fold(raw_table_text(t)) for t in self.tables)

    def scores(self, question):
        return linear_kernel(self.vec.transform([fold(question.lower())]), self.matrix)[0]


class LemmaRetriever(_Base):
    def __init__(self, tables=None):
        super().__init__(tables)
        self.places = Places(self.tables)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(
            " ".join(lemmas(lemma_table_text(t, self.places.names))) for t in self.tables
        )

    def scores(self, question):
        s = linear_kernel(self.vec.transform([" ".join(lemmas(question))]), self.matrix)[0]
        return self.places.boost(question, s, PLACE_BOOST["lemma"])


def embedding_texts(tables, model_name=EMBED_MODEL):
    """(texts to embed, cache file) for these tables. scripts/embed_on_gpu.sh uses the same pair,
    so vectors computed on another machine land exactly where EmbeddingRetriever looks."""
    texts = [table_text(t) + (" | " + t["title_en"] if t.get("title_en") else "") for t in tables]
    key = hashlib.sha1((model_name + "\n".join(texts)).encode()).hexdigest()[:12]
    return texts, DATA / f"emb_{model_name.split('/')[-1]}_{key}.npy"


class EmbeddingRetriever(_Base):
    """Table vectors are computed once and cached in data/ (keyed by catalog content).
    On a CPU (a Raspberry Pi 5) that takes ~1.5 h for bge-m3; scripts/embed_on_gpu.sh does it in minutes."""

    def __init__(self, tables=None, model_name=EMBED_MODEL):
        super().__init__(tables)
        from sentence_transformers import SentenceTransformer  # heavy import; only when used

        self.model = SentenceTransformer(model_name)  # GPU when there is one (DGX, LUMI), else CPU
        texts, cache = embedding_texts(self.tables, model_name)
        if cache.exists():
            self.matrix = np.load(cache)
        else:
            self.matrix = self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
            np.save(cache, self.matrix)

        self.model_name = model_name
        self._encode = functools.lru_cache(maxsize=4096)(self._encode_cached_on_disk)

    def _encode_cached_on_disk(self, question):
        """Question vector, from data/question_vectors.sqlite if seen before (any process, any run);
        embedding a question on a CPU takes ~1 s. SQLite because the server and eval runs
        may write at the same time."""
        with sqlite3.connect(QUESTION_VECTORS, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS qvec (model TEXT, q TEXT, vec BLOB, PRIMARY KEY (model, q))")
            row = db.execute("SELECT vec FROM qvec WHERE model=? AND q=?", (self.model_name, question)).fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
            vec = self.model.encode([question], normalize_embeddings=True)[0].astype(np.float32)
            db.execute("INSERT OR REPLACE INTO qvec VALUES (?, ?, ?)", (self.model_name, question, vec.tobytes()))
            return vec

    def scores(self, question):
        return self.matrix @ self._encode(question)


class FusionRetriever(_Base):
    """Reciprocal rank fusion: a table ranked high by either retriever ends up high. The place boost is
    applied again after fusion, else raw + embed outvote lemma on "í Kópavogi" questions."""

    def __init__(self, retrievers, c=60):
        super().__init__(retrievers[0].tables)
        self.retrievers, self.c = retrievers, c
        self.weights = np.ones(len(self.tables))  # members already applied archive weights
        self.places = Places(self.tables)

    def scores(self, question):
        total = np.zeros(len(self.tables))
        for r in self.retrievers:
            s = r.scores(question) * r.weights
            rank = np.empty(len(s))
            rank[np.argsort(-s)] = np.arange(len(s))
            total += 1 / (self.c + rank)
        return self.places.boost(question, total, PLACE_BOOST["fusion"])


def places_of(retriever):
    """The Places of a retriever chain (bilingual -> checked -> fusion), or None."""
    r = retriever
    while r is not None and not hasattr(r, "places"):
        r = getattr(r, "base", None)
    return getattr(r, "places", None)


def agreement(retriever, question, table):
    """How many of the fused searches (raw, lemma, embed) put this table first: 0-3, None if the retriever has no
    members. On dev, a #1 that all three agree on was right 87% of the time, one they all differ on 25%
    (eval/experiments/pick_strategies.py): a confidence signal computed in code, not asked of the model."""
    r = retriever
    while not hasattr(r, "retrievers") and hasattr(r, "base"):
        r = r.base  # bilingual -> checked -> fusion
    members = getattr(r, "retrievers", None)
    if not members:
        return None
    q = correct_question(question)
    return sum(bool(top) and top[0][0]["id"] == table["id"] for top in (m.search(q, k=1, correct=False) for m in members))


class BilingualRetriever:
    """An English question is also searched as its Icelandic translation (translate.counterpart) and the two
    rankings are fused (reciprocal rank fusion). Measured 31 Oct with 'checked' as the base: English dev 39/64/74
    -> 46/70/82 (top1/top3/top8, equal weights; 0.8 for the translation gave 78 top8); the other way round,
    Icelandic questions lost (51/71/86 -> 47/71/84), so they
    are searched as they are. Without a model, or when translation fails, it is exactly the base retriever."""

    def __init__(self, base, pool=20, c=60, own_weight=1.0, other_weight=1.0, english_only=True):
        self.base, self.pool, self.c, self.tables = base, pool, c, base.tables
        self.own_weight, self.other_weight, self.english_only = own_weight, other_weight, english_only

    def search(self, question, k=8, correct=None):
        own = self.base.search(question, k=max(k, self.pool), correct=correct)
        if self.english_only and not is_english(question):
            return own[:k]
        try:
            from .translate import counterpart

            other_q = counterpart(question)
        except Exception:  # noqa: BLE001 - no model reachable, a bad reply: one language is still a search
            log.info("no translation for %r", question, exc_info=True)
            return own[:k]
        if not other_q or other_q.strip().lower() == question.strip().lower():
            return own[:k]
        other = self.base.search(other_q, k=max(k, self.pool), correct=correct)
        fused, tables = defaultdict(float), {}
        for weight, ranking in ((self.own_weight, own), (self.other_weight, other)):
            for rank, (t, _) in enumerate(ranking):
                key = (t["id"], t["path"])
                fused[key] += weight / (self.c + rank)
                tables[key] = t
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(tables[key], fused[key]) for key in best]


def rrf(rankings, k=8, c=60):
    """Reciprocal rank fusion of result lists [(table, score), ...]: each table gets sum(1 / (c + rank))."""
    fused, tables = defaultdict(float), {}
    for ranking in rankings:
        for r, (t, _) in enumerate(ranking):
            key = (t["id"], t["path"])
            fused[key] += 1 / (c + r)
            tables[key] = t
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(tables[key], fused[key]) for key in best]


def build(name, tables=None, _cache={}):
    """Retriever by name; built once per process (the embedding one takes a while)."""
    if name not in _cache:
        tables = tables if tables is not None else load_catalog()
        if name == "raw":
            _cache[name] = RawRetriever(tables)
        elif name == "lemma":
            _cache[name] = LemmaRetriever(tables)
        elif name == "embed":
            _cache[name] = EmbeddingRetriever(tables)
        elif name == "embed2":  # the second embedding model (EMBED_MODEL_2)
            _cache[name] = EmbeddingRetriever(tables, model_name=EMBED_MODEL_2)
        elif name == "fusion":  # exact values (raw), inflection + synonyms (lemma), meaning (embed, and embed2 if set)
            members = [build("raw", tables), build("lemma", tables), build("embed", tables)]
            _cache[name] = FusionRetriever(members + ([build("embed2", tables)] if EMBED_MODEL_2 else []))
        elif name == "checked":  # fusion, then tables lacking the asked place/year move down (checker.py)
            from .checker import CheckedRetriever
            _cache[name] = CheckedRetriever(build("fusion", tables))
        elif name == "bilingual":  # checked, on the question and on its translation (needs the LLM)
            _cache[name] = BilingualRetriever(build("checked", tables))
        else:
            raise ValueError(f"unknown retriever {name!r}")
    return _cache[name]


RETRIEVERS = ["raw", "lemma", "embed", "fusion", "checked"]
DEFAULT = "bilingual"  # checked + English questions also searched in Icelandic: test_en 31/53/69 -> 38/61/76 (31 Oct)
