"""The chat as a conversation (31 Oct): follow-up questions, and messages that mix Icelandic and English.

  standalone(message, previous) -> the question to answer: "En 2024?" after "Hvað búa margir í Kópavogi?" becomes
      "Hvað bjuggu margir í Kópavogi árið 2024?" (one LLM call, only for short or continuing messages)
  mixed(message) -> True when the message mixes Icelandic and English question words ("How many people búa í
      Reykjavík?"): the chat then asks the user to choose a language instead of guessing

Chat only: the API and the evaluations answer each question as asked.
"""
import re

from . import llm

FOLLOWUP_START = re.compile(r"^\s*(en|og|hvað með|hvad med|sama|líka|lika|but|and|what about|how about|same|also)\b",
                            re.I)
MAX_FOLLOWUP_WORDS = 4  # a message this short is read with the question before it
REWRITE_SYSTEM = (
    "Þú færð fyrri spurningu notanda og ný skilaboð frá honum. Ef nýju skilaboðin halda áfram með fyrri "
    "spurninguna (t.d. 'En 2024?', 'og á Akureyri?', 'What about men?'), skrifaðu þau sem eina sjálfstæða "
    "spurningu á tungumáli nýju skilaboðanna, með því sem vantar úr fyrri spurningunni (efni, stað, tímabil, hóp). "
    "Ef nýju skilaboðin eru sjálfstæð spurning, skilaðu þeim óbreyttum. Bættu engu öðru við. "
    'Svaraðu eingöngu með JSON: {"question": "..."}'
)
# question and function words only: an English noun in an Icelandic question ("inflation rate") or an Icelandic place
# in an English one ("in Kópavogur") is normal, and is not a mix
ICELANDIC = set("hvað hvad hve hversu margir margar mörg morg voru eru var búa bua bjuggu árið arid hvert hvaða "
                "hvada með med frá fra fyrir og eða eda".split())
ENGLISH = set("how many much what which were was did does the people live lived there iceland".split())
UNDERSTOOD = {"is": "Skilið sem", "en": "Understood as"}
CHOOSE = ("Spurningin blandar saman íslensku og ensku. Á hvoru málinu viltu svarið? Spyrðu aftur á íslensku eða "
          "ensku.\n\nYour question mixes Icelandic and English. Which language do you want the answer in? Please ask "
          "again in Icelandic or in English.")


def looks_like_followup(message):
    return bool(FOLLOWUP_START.match(message)) or len(re.findall(r"\w+", message)) <= MAX_FOLLOWUP_WORDS


def standalone(message, previous):
    """The message as a question that stands alone; the message itself when there is nothing to continue."""
    if not previous or not looks_like_followup(message):
        return message
    try:
        q = llm.chat_json(REWRITE_SYSTEM, f"Fyrri spurning: {previous}\nNý skilaboð: {message}").get("question")
    except (ValueError, AttributeError):  # no usable reply: answer the message as written
        return message
    return q.strip() if isinstance(q, str) and q.strip() else message


def mixed(message):
    words = set(re.findall(r"\w+", message.lower()))
    return bool(words & ICELANDIC) and bool(words & ENGLISH)
