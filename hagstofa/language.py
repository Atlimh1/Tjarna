"""Which language a question is in, without a model: writing system, letters only some languages use, and
short common words. Icelandic and English go straight to the pipeline; any other language is translated
(hagstofa/translate.py).

  detect(text) -> "is", "en" or another ISO 639-1 code ("pl", "lt", "uk", "zh", ...)

Icelandic is the default: a question typed without accents ("hvad bua margir") has no Icelandic letters and
few known words, so another language must win clearly. Measured by eval/lang_eval.py: every question in our
Icelandic/English sets must come out "is" or "en".
"""
import re

from .retrieve import is_english

NAMES = {"is": "Icelandic", "en": "English", "pl": "Polish", "lt": "Lithuanian", "lv": "Latvian",
         "uk": "Ukrainian", "ru": "Russian", "de": "German", "es": "Spanish", "fr": "French", "da": "Danish",
         "no": "Norwegian", "sv": "Swedish", "pt": "Portuguese", "it": "Italian", "ro": "Romanian",
         "nl": "Dutch", "vi": "Vietnamese", "tl": "Filipino", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
         "ar": "Arabic", "fa": "Persian", "th": "Thai", "el": "Greek", "he": "Hebrew", "hi": "Hindi"}

# writing systems: (range, language); Cyrillic is decided by letters below
SCRIPTS = [("぀-ヿ", "ja"), ("가-힯", "ko"), ("一-鿿", "zh"), ("؀-ۿ", "ar"),
           ("฀-๿", "th"), ("Ͱ-Ͽ", "el"), ("֐-׿", "he"), ("ऀ-ॿ", "hi")]
CYRILLIC = re.compile("[Ѐ-ӿ]")
UKRAINIAN = re.compile("[іїєґІЇЄҐ]")
PERSIAN = re.compile("[پچژگک]")

# letters that point to one language (each hit counts 2); Icelandic þ/ð decide on their own
LETTERS = {
    "pl": "ąćęłńśźż", "lt": "ėįųū", "lv": "āēģīķļņ", "de": "ßü", "es": "ñ¿¡", "fr": "çœèêëîôùûÿ", "da": "ø",
    "no": "ø", "sv": "", "pt": "ãõç", "it": "ìò", "ro": "ăîșțşţ", "vi": "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđơư",
}
WORDS = {
    "is": "hvað hvad hve hversu margir margar mörg morg árið arid ári ari voru eru hver hvert hvaða hvada fjöldi "
          "fjoldi eftir frá fra með med íbúar ibuar hlutfall búa bua bjuggu fæddust faeddust landinu íslandi "
          "islandi á í og til um af var er við vid a i ad að eda eða ur úr thad það sem hja hjá hins arsins "
          "folk fólk marga",
    "pl": "ile ilu jak jaka jaki jakie był była było były roku osób osob mieszka mieszkało się czy w we na do z i",
    "lt": "kiek koks kokia kokie buvo gyvena gyveno metais m ir į kas yra",
    "lv": "cik kāds kāda bija gadā gada dzīvo un uz ir",
    "de": "wie viele war waren ist im der die das und jahr hoch leben lebten nach wurden am in",
    "es": "cuántos cuántas cuantos cuantas cuál cual fue en el la de los las y qué personas viven año",
    "fr": "combien quel quelle était etait le la les et des est personnes vivent sont en de",
    "da": "hvor mange hvad var og der er bor boede blev til den af mennesker i island",
    "no": "hvor mange hva var og er bor bodde ble til på mennesker i island",
    "sv": "hur många vad var och är bor bodde till på människor föddes i island",
    "pt": "quantas quantos qual quais foi foram para na no em de da do e pessoas vivem",
    "it": "quante quanti quale qual era il la di e nel nella della sono persone vivono in",
    "ro": "câți câte cati care a fost în și de la oameni",
    "nl": "hoeveel wat was in de het een en mensen wonen woonden",
    "vi": "có bao nhiêu người ở năm vào là của",
    "tl": "ilan ang mga sa noong nakatira ng ay",
}
WORD_SETS = {lang: set(words.split()) for lang, words in WORDS.items()}
MARGIN = 1  # points another language needs above Icelandic (set on dev_typo: no-accent Icelandic stays "is")
TOKEN = re.compile(r"[^\W\d_]+")


def _english(word):
    from .guard import english_vocabulary  # guard imports this module

    return len(word) > 3 and word in english_vocabulary() and not any(word in ws for ws in WORD_SETS.values())


def scores(text):
    """Points per Latin-script language: 1 per common word, 2 per letter only that language uses. English words
    longer than 3 letters count for no language: Icelanders mix them in ("hvad var inflation rate i september")."""
    low = text.lower()
    words = [w for w in TOKEN.findall(low) if not _english(w)]
    out = {lang: sum(w in ws for w in words) for lang, ws in WORD_SETS.items()}
    for lang, letters in LETTERS.items():
        out[lang] += 2 * sum(low.count(c) for c in letters)
    return out


def detect(text):
    for chars, lang in SCRIPTS:
        if re.search(f"[{chars}]", text):
            return "fa" if lang == "ar" and PERSIAN.search(text) else lang
    if CYRILLIC.search(text):
        return "uk" if UKRAINIAN.search(text) else "ru"
    if is_english(text):  # first: English questions name Icelandic places ("in Hafnarfjörður", "in Garðabær")
        return "en"
    if re.search("[þðÞÐ]", text):
        return "is"
    s = scores(text)
    best = max((lang for lang in s if lang != "is"), key=s.get)
    # another language only when it clearly beats Icelandic (no-accent Icelandic shares words with Danish)
    return best if s[best] >= 2 and s[best] >= s["is"] + MARGIN else "is"


# --- several questions in one message ----------------------------------------------------------------------
QUESTION_END = re.compile(r"(?<=[?？؟])[ \t]+")  # after a question mark, on the same line
LIST_MARK = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+")
MIN_WORDS = 4  # a shorter part in the same language is a follow-up ("En í Reykjavík?") and stays with the last
MAX_QUESTIONS = 5


def split_questions(text):
    """A message -> the separate questions in it, each answered on its own (and in its own language).
    One question stays one: "Hvað búa margir á Íslandi? En í Reykjavík?" (follow-up), "Hver var verðbólgan?
    Þ.e. 12 mánaða breyting" (a clarification without a question mark), "í Reykjavík og Kópavogi" (one sentence:
    the pipeline's decompose() handles that). Lines and list items ("1. ...", "- ...") are separate questions."""
    lines = [line for line in text.splitlines() if line.strip()]
    listed = sum(bool(LIST_MARK.match(line)) for line in lines) >= 2
    pieces = []  # (text, starts a new line)
    for line in lines:
        line = LIST_MARK.sub("", line) if listed else line
        for i, piece in enumerate(QUESTION_END.split(line.strip())):
            pieces.append((piece.strip(), i == 0))
    out = []
    for piece, new_line in pieces:
        follow_up = len(TOKEN.findall(piece)) < MIN_WORDS and out and detect(piece) == detect(out[-1])
        clarification = not new_line and not piece.endswith(("?", "？", "؟"))
        if out and (follow_up or clarification):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out
