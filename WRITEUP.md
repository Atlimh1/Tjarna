# Spurðu Hagstofuna: write-up

Track 2 (direct PX-API): a chatbot that answers questions about Statistics Iceland's data in Icelandic or
English, with the numbers read from Statistics Iceland's own tables, the arithmetic done in code, and every
table cited. This file is the record of what we changed, what it did to the score, what blocked us and what
we would do next. Numbers come from runs, not estimates; each change says which run measured it.

## How it answers

```text
question ─► split in ≤2 sub-questions (LLM)                      "per inhabitant" → quantity + population
         ─► per sub-question: table search (no LLM) → top 8         raw + lemma (BÍN) + bge-m3, fused, then checked
         ─► LLM picks one table (forced: its refusal → search #1)
         ─► fallback chain until a result holds a number:
              tables (picked, then next by rank) × editions (question's language first) × sources (data release, PX-Web)
         ─► LLM chooses the values (codes checked against the table)
         ─► LLM names one operation; code computes it (also across the two sub-questions)
         ─► LLM writes the answer; code adds any computed result it left out, and every source
```

## Scores (the organizers' `gagnvist evaluate`, their 60 questions, their scoring code unchanged)

| Run | When | What changed before it | Easy | Medium | Hard | EN | IS | **All** |
|---|---|---|---|---|---|---|---|---|
| Starter kit | (organizers) | their baseline, 52 questions | | | | | | **7 of 52** |
| 1 | 31 Oct 01:29 | first full run: our search + pipeline, local data, English editions | 15/20 | 6/20 | 6/20 | 12 | 15 | **27/60** |
| 2 | 31 Oct 01:37 | forced pick, calculation across sub-questions, results kept in the answer | 16/20 | 10/20 | 9/20 | 17 | 18 | **35/60** |
| 3 | 31 Oct 01:55 | picker rules for look-alike tables, per-capita split (see caveat) | 16/20 | 14/20 | 10/20 | 22 | 18 | **40/60** |
| Starter kit, our model | 31 Oct 13:47 | the organizers' own pipeline with our Qwen3-30B: a fair baseline | 6/20 | 2/20 | 0/20 | 2 | 6 | **8/60** |
| 4 | 31 Oct 14:08 | merged main (guard, languages, number check), pipeline only | 17/20 | 13/20 | 11/20 | 22 | 19 | **41/60** |
| 4, front door | 31 Oct 14:10 | the same, through `app.ask` as users get it: guard, LLM safety check, number check | 17/20 | 10/20 | 9/20 | 17 | 19 | **36/60** |
| 6, front door | 31 Oct 16:30 | Gemma 4 31B (chosen in run 5), bilingual search, calculation fixes | 16/20 | 18/20 | 13/20 | 21 | 26 | **47/60** |
| 7, front door | 31 Oct 16:58 | the picker sees each table's unit; number check reads full-precision values | 18/20 | 18/20 | 16/20 | 24 | 28 | **52/60** |
| **8, front door** | 31 Oct 17:00 | value choices read by label too; long decimals rounded (base bge-m3, the demo setup) | 19/20 | 18/20 | 17/20 | 26 | 28 | **54/60** |
| 8, fine-tuned search | 31 Oct 17:00 | the same with our fine-tuned bge-m3 | 20/20 | 18/20 | 18/20 | 29 | 27 | **56/60** |
| 9, front door | 31 Oct 17:15 | fixes from the demo dry run (stale tables, assumptions line, confidence line) | 19/20 | 18/20 | 18/20 | 26 | 29 | **55/60** |
| 10, front door | 31 Oct 17:35 | 2024 revision tables as archive | 19/20 | 18/20 | 17/20 | 26 | 28 | 54/60 |
| **final, front door** | 31 Oct 20:40 | the evening's fixes, four-way search, 12 candidates: **the code we demo** (see "Toward 60/60") | 20/20 | 20/20 | 20/20 | 30 | 30 | **60/60** |

Times are JURECA's clock (Central European Time, an hour ahead of Iceland). Run 4: `scripts/jureca_eval.sh` at
main `1eda47d`, log `eval/results/jureca-20261031-140603.log` on JURECA.

Model: Qwen3-30B-A3B-Instruct-2507 on JURECA (4 × A100 40 GB, vLLM 0.32 in our own Python environment), about
1–3 seconds per question. Search only (no LLM), right table first: ours 30/60, the starter's 6/60
(`eval/experiments/gagnvist_retrieval.py`).

**Honest caveat.** From run 3 on, the picker rules were prompted by failures on these same 60 questions. We
kept them general (no table numbers, no topics: rules any statistician would state), but scores on these 60
overstate how well it does on new questions.

## Beyond the 60 (run 4, 31 Oct 14:06-14:17, through the front door)

| What | Result |
|---|---|
| Our own frozen set, 41 questions no rule was tuned on (`eval/questions_new.jsonl`) | **13/41** (easy 10/24, medium 2/14, hard 1/3; EN 5/13, IS 8/28) |
| 16 questions Statistics Iceland can't answer (`eval/decline_eval.py`) | declined **3/16** (2 off-topic, 1 personal); 13 answered from some table |
| Guard, rules + LLM safety check, 239 bad inputs | 233/239 stopped, off-topic now 16/16 stopped; but 5 of ~400 real questions refused as off-topic (radio content shares, design firms) |
| 19 other languages, 2 questions each, vs the English original | right table 37/38, same numbers 19/38; the answer came back translated in only 4/38 (the translation changed a number, so the English answer was shown) |

What it shows:
- **The front door costs 5 of the 60, all English calculation questions** (shares, per 1,000, per inhabitant). In
  those answers the model did the arithmetic itself, correctly, because the code's calculation step produced no
  calculation; the number check then saw a number in no source and replaced the answer with the rows. The fix is
  upstream: the calculation step must fire for share and per-1,000 questions (it often doesn't, in either language).
- **On new questions the score falls from 40/60 to 13/41.** This is the honest number: the picker rules were shaped
  on the 60. 24 of the 28 misses state a different number than expected; the per-question causes are not analysed yet.
- **Forced pick answers what it should decline**: 13 of the 16 no-data questions got an answer from some table.

## Run 5: the fixes, and the model choice (31 Oct 15:27-15:50, main `c461271`, same code for every model)

After run 4: the calculation planner's "two separate numbers -> none" rule was reworded and backed by a plan in
code (shares, per 1,000, per inhabitant); the number check accepts correct two-number arithmetic on the rows;
a table without the question's year is skipped; the LLM check's topic list names firms, sectors and media.
(A "nodata" class in the LLM check was tried in between and reverted: to the models, 2026 is "the future",
and it refused 110 of 120 real answers.) Every model ran the same `scripts/jureca_eval.sh` on JURECA.

| Model (open weights) | The 60, pipeline | **The 60, front door** | **Our 41 new, front door** | of which Icelandic | Declined of 16 | Real questions refused | Off-topic stopped | Other languages: answered in that language | s/question |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (other session, earlier code) | 12 | | | | | | | | 3.8 |
| Qwen3-30B-A3B-Instruct-2507 | 42 | 42 | 14 | 8/28 | 3 | 0/396 | 15/16 | 4/38 | 1.4 |
| Qwen2.5-72B-Instruct (AWQ) | 47 | **49** | 10 | 6/28 | 4 | 0/396 | 16/16 | 9/38 | 5.8 |
| **Gemma 4 31B** (Apache-2.0) | 45 | 45 | **21** | **14/28** | 3 | 2/396 | 16/16 | **20/38** | 5.0 |

What it shows:
- **The front door no longer costs points** (Qwen3-30B: 36 -> 42 = its pipeline score): the calculation fix works.
- **The best score on the 60 is not the best model.** The 72B leads on the 60 our rules were shaped on and is
  last on the 41 new questions; Gemma 4 is best on new questions, twice as good in Icelandic, and the only one
  that answers most foreign-language questions in their own language. **Gemma 4 31B is our model.**
- Declines are still the weak spot (3-4 of 16 with every model).

## Runs 6-8: Gemma 4, and what its misses taught (31 Oct 16:00-17:10, front door, `eval/confidence_eval.py`)

Run 6 (main `3485590`, Gemma 4, bilingual search): **47/60** (easy 16, medium 18, hard 13; EN 21, IS 26) and
**20/41**. Reading every miss gave four causes, each general, each fixed in code with tests:

| Cause | Seen in run 6 | Fix |
|---|---|---|
| The number check rejected values the model copied with all their decimals ("38,823235", "120.669309808456"): it knew 0-3 decimals | 10 answers replaced (6 of the 60, 4 of the 41) | values as stored and rounded to 0-12 decimals, and scaled by the table's unit ("16 milljónir" from thousand krónur) (`cf6cd2c`) |
| Look-alike tables: "Iceland's GDP" from THJ01000 (indices, volume growth) instead of THJ01102 (million krónur) | 6 of the 60 | the picker sees each table's unit, and prefers amounts over indices unless an index is asked for (`3214c3f`) |
| The model keyed its value choices as the prompt shows them ({"tími": ["2026M08"]}, {"Rate": [...]}); only codes were read, so the latest month and the first measure were used | found tracing the wage index for August 2026 and unemployment in Q2 2026 | `chosen()` reads codes, labels, English labels and the time marker; unknown names are still dropped (`28ee3a8`) |
| The organizers' scorer reads a number only to the digits it shows, against a key of at most 4 decimals: 38.823235 is not 38.8232 to it, 38.82 is | 2 of the 41, right table and value | numbers with 5+ decimals are rounded to 2-4 (`3adccad`) |

Run 7 (`3214c3f`, the picker change alone): **52/60** and **22/41**: the unit rule fixed all 6 GDP questions and
generalised to the unseen questions. Run 8 (`3adccad`, all four), with both search models side by side:

| Run 8, Gemma 4, front door | The 60 | **Our 41 unseen** | of which Icelandic | Confidence: right when high / medium / low |
|---|---|---|---|---|
| plain bge-m3 (**the demo**) | 54 | **25** | 18/28 | **10/10 (100%)** / 21/25 (84%) / 48/66 (73%) |
| our fine-tuned bge-m3 | **56** | 24 | 17/28 | 7/10 (70%) / 29/34 (85%) / 44/57 (77%) |

From run 6 to run 8: **47 -> 54 of the 60, 20 -> 25 of the 41.** The two search models tie end to end (79 vs 80
of 101): the fine-tuned one finds the long-run wage index (LAU04000) but loses two CPI questions to a look-alike
(VIS02500). We demo plain bge-m3 because its confidence line is the one that holds: when all three searches agree
it was right 10 times of 10, and the levels order as they should. The calibration is in `data/confidence.json`.

Two of the four causes surfaced on our frozen 41, so from run 7 on the 41 is less clean as a held-out set; every
fix is about format or reading, none names a table or a topic. The 4 misses left on the 60 (fine-tuned) and the
6 (plain) are look-alike tables: the wage index from 2015 instead of 1989, a CPI breakdown, a sex breakdown.

**Dry run of the demo (17:10-17:40).** The demo questions through `app.respond`, as the chat shows them, found
what no score did:
- "Hvað búa margir á Íslandi?", the first example button, answered from HEI11100 (titled just "Mannfjöldi",
  last updated 2019: mean population 2017). Fixes: the picker sees each table's last update and, with no period
  asked, tables 3+ years older than the newest candidate are left out of its list (`fresh_first`); the 2024
  method-revision tables (population overestimate) count as archive in the search (dev 138 -> 139 right first,
  nothing lost; raising the overview folders instead cost 8 and was dropped).
- "How many elves live in Iceland?" said the data does not answer, then "Assuming: Sex Total, ...": no
  assumptions line under an answer that states no number.
- The second opinion compared "1841" from a title ("1841-2026"), not the population: year ranges are skipped.
- The confidence line read "low (right 73% ...)" under right answers: it now leads with the measured rate,
  "Confidence: 73% (answers like this were right in 48 of 66 tests)".

| Run (Gemma 4, front door, plain bge-m3) | The 60 | Our 41 | Confidence: right when high / medium / low |
|---|---|---|---|
| 9 (`e5e7c53`, the dry-run fixes) | 55 | 25 | 11/11 / 20/24 / 49/66 |
| **10** (`04d8361`, + revision tables as archive: **the demo**) | **54** | **26** | 11/12 (92%) / 20/24 (83%) / 49/65 (75%) |

Run 9 and 10 both score 80 of 101; one wage-index question moved between them (the model's choice varies run to
run), one unseen population question was fixed. `data/confidence.json` holds run 10.

## Toward 60/60 (31 Oct evening, 18:00-21:00 JURECA time)

**A second unseen set first.** Fixing the 41's misses makes the 41 no longer unseen, so before any fix a separate
agent, shown only the catalog and the first set's builder, wrote 45 new questions on 46 tables no set uses
(`eval/build_new_set2.py`, `questions_new2.jsonl`; 27 IS / 18 EN; 18 easy, 17 medium, 10 hard). Numbers are read
from the release in code. **Set 2 is frozen**: no change was made because of its misses (decided with the team).

**Answer keys accept equal tables, verified.** The organizers' key accepts every table that holds exactly the
expected number; ours listed one. A table was added only where the rows an answer read from it hold the part's
exact value (or values summing to it), checked in the data: 7 questions in set 1, 8 in set 2 (`KEY_FIXES`).

**What changed, each general and tested (338 tests in all):**

| Change | Why (the miss that showed it) |
|---|---|
| Formula engine (`calc.py`): the model writes `(heild - hofudb) / heild * 100` or `sum(eldri) / allir * 100` over named rows; code evaluates a whitelisted syntax tree (no `eval`) | one operation per question could not do (all minus capital) / all, or the share of ages 68+ |
| Self-check: the model is shown the rows read and says whether they answer the question; if not, the next table | wrong measure (activity rate for unemployment), wrong place, an index for an amount |
| The picker sees each table's measures (`Hlutfall, %: Atvinnuþátttaka, ... Atvinnuleysi`), the question's values first | look-alikes with the same title; LAN10102's first six values are cattle, its sheep later |
| Value choices read in the prompt's own formats: `code (label)`, `code=label` | the right code for the National Church was dropped, so the total was used |
| Villages: a town that is no municipality (Vík, not Mýrdalshreppur) gets a note and the towns tables as candidates; short names and missing accents keep their value in the list | "How many people live in Vik?" |
| Rule 6: the 1 January population over the yearly average, also as the base for per-head figures (as the organizers' questions do) | Húsavík "núna" answered from the 2025 average |
| A per-head question with a direct table (`Landsframleiðslan á mann`) is not split | GDP per head computed from two tables instead of read |
| Four-way search: our fine-tuned bge-m3 joins the fusion (`EMBED_MODEL_2`) | dev 51/72/87 -> 57/78/91, English dev 46/70/83 -> 54/78/87 |
| Synonyms: kjörsókn/turnout -> kosningaþátttaka, heita/names -> eiginnöfn, sauðfé/cattle -> búpeningur | the tables use words people don't |
| The self-check is told today's date | it refused 1 January 2026 as "a forecast, not now" |
| Accent restoration for questions typed without Icelandic letters (Atli's session) | no-accent questions 43 -> 50% right table first |

Scores through the front door (Gemma 4), "other table" counted only where the key now accepts it:

| Run | Code | The 60 | Set 1 (41) | Set 2 (45, frozen) |
|---|---|---|---|---|
| 10 (start of the evening) | `04d8361` | 54 | 28 | 29 |
| v2 | `479aca7`, all of the above but synonyms and the date, four-way | 58 | 35 | 35 |
| v4 | `1dea23e`, + Atli's accents, synonyms | 57 | 35 | 35 |
| final, 8 candidates | `9aabec1`, + village values, the date in the check | 58 | 36 | 36 |
| **final, 12 candidates** | the same, the picker sees 12 tables (`PICK_K`, now the default) | **60** | **37** | **36** |

The fixes carried over to questions nobody looked at: **set 2 rose from 29 to 36** without a single change made for
it. 12 candidates instead of 8 put the 1989 wage index (ranked 7th-12th by search) in front of the picker: 60/60 on
the organizers' questions, nothing lost on either unseen set, 4.0 -> 4.1 s per question. Confidence (final code,
101 questions with known answers): right when high 16/16, medium 25/26 (96%), low 56/59 (95%), in
`data/confidence.json`.

Still wrong on set 1 (4): the name Jón (the right table, the wrong number), weekly hours "now" (a look-alike), the
Alþingi turnout change (the right table, a wrong number), and Árborg's turnout, where two official figures exist
(63.7% by age and municipality, 63.8% votes cast / registered) and our key holds one. Set 2's misses (9) are mostly
hard per-head questions; set 2 stays frozen, so they are reported, not tuned.

## The whole system (31 Oct, night, 21:40-23:00 JURECA time)

Measured on the final code, Gemma 4, through the front door:

| What | Before | **After** | How |
|---|---|---|---|
| Unpublished questions declined (`eval/decline_eval.py`, 16) | 3 | **15** | when no tried table passes the self-check, "I found no data" instead of the closest table (`DECLINE_UNCHECKED`) |
| The team's own 42 questions (`eval/real2_eval.py`, frozen) | 23 | **31** | the same: declines 9 -> 14 of 16, partial 7 -> 11 of 13, answer 7 -> 6 of 13 |
| Other languages, same numbers as the English original (`eval/lang_eval.py`, 19 languages x 2) | 19/38 | **37/38** | the translation check read "January 1, 2026" as one number (12026), so correct translations fell back to English |
| Other languages, answered in the user's language | 22/38 | **36/38** | the same fix |
| The 60 / the 41 / set 2 with declining on | 60 / 37 / 36 | 60 / 37 / 35 | nothing lost; one set-2 answer varies between runs (VIS01106), not a decline |

Declining has one cost on the team set: "hvað eru margir þyngmenn á alþingi?" was answered right before and is now
declined. Four "answer" questions declined now (immigrants, coffee imports, data centres' electricity, hospital
beds) were already wrong, from unrelated tables; "no data" is the honest reply there. Safety on the same code:
2 of 396 real questions refused; every bad-input category stopped (abuse 10/10, binary 16/16, gibberish 21/21,
greetings 18/18, ...).

Also new in the chat (tested in code, 359 tests): a short or continuing message ("En 2024?", "what about men?") is
rewritten into a whole question with the one before it and shown as "Skilið sem: ..."; a message that mixes
Icelandic and English question words gets a reply asking which language to answer in (the API answers as asked);
whole numbers are grouped (4.387 / 4,387; years, codes and decimals untouched); the search is warmed up when the
server starts, so the first question no longer takes ~30 s.

## From the team's list of ideas (31 Oct, near midnight; backend only, the UI untouched)

Taken from "Hagstofan: improvements to consider" (Atli), built in `hagstofa/` so the UI can be changed in
parallel; everything new comes back as extra fields on the result, the API and the scorer answer as before.

- **B, a polite "no" with questions we can answer** (`suggest.py`): after "no data", 2-3 related questions in the
  user's language, for the search's candidate tables: the closest of the 4,580 questions written for the
  fine-tuning (`data/sample_questions.jsonl`), else one built from the table's title. In the text and as
  `suggestions`.
- **E, what a vague question left open** (`clarify.py`): for the period and the place that code filled (the
  "Assuming" line), up to two questions with the default first and up to 4 options, as `clarify`, for a client to
  offer as buttons. The answer is never held back.
- **A, names**: "Hve margir heita Jón?" -> 5.052, "How many people in Iceland are named Jón?" -> 5,052, "Guðrún"
  -> 4.472, "the most common male name" -> Jón, all from the names tables (1 January 2023). Our set-1 version asks
  for 1 January 2025 explicitly; the table is a 2023 snapshot and the self-check refuses it, which is defensible.
- **Largest / smallest** ("which year had the most births") was built and **not kept**: with it described in the
  calculation prompt, a % change question of the 60 got a wrong number (60 -> 59) and set 2 lost one; no question
  used it. "Which year had the most" stays a known limit.
- The self-check accepts a table's newest figures when the question names no period.

Measured with Gemma 4 through the front door: the 60 60/60, the 41 37, set 2 35, declined 15 of 16, the team set
31 of 42: nothing lost. The five answerable team questions still declined (coffee imports, immigrants, data
centres' electricity, hospital beds, the misspelt "þyngmenn") are search misses: the right tables never reach the
candidates ("flutt inn", imported, reads as migration), and the self-check rightly refuses what does.

Two search fixes followed (`retrieve.py`): "flytja inn / út" now means trade (innflutningur / útflutningur), not
migration, and "innflytjendur / immigrants" finds MAN43000 (immigrants are a value of "bakgrunnur"). On every
question in our sets they touch, the right table moved up or stayed (export questions 2->1, 4->2, 4->2; immigrant
questions 3->1, 2->1); "hvað er flutt inn mikið af kaffi á ári?" is now answered (2,504 tonnes in 2025, UTA06202).
The 60 60/60, the 41 37, set 2 35 after both.

## Icelandic and English, and spelling (31 Oct afternoon)

English search had been measured on the organizers' 30 English questions only. `eval/build_english_set.py` had
the model translate dev and test into English, same expected tables (227 and 536 questions kept), so English is
now measured like Icelandic. Table search, right table first / top 3 / top 8:

| | Icelandic dev (270) | English dev (227) | English test (536, frozen, run once) |
|---|---|---|---|
| `checked` (before) | 51 / 71 / 86 | 39 / 64 / 74 | 31 / 53 / 69 |
| **`bilingual`** (the default now) | 51 / 71 / 86 (unchanged) | **46 / 70 / 82** | **38 / 61 / 76** |

`bilingual`: an English question is also searched as its Icelandic translation (one cached LLM call) and the two
rankings are fused, so English borrows the Icelandic search: BÍN word forms, synonyms, place checks. Run the other
way, Icelandic questions lost (51/71/86 -> 47/71/84), so they are searched as they are. Equal weights beat 0.8 for
the translation (top 8: 82 vs 78). The gap to Icelandic shrinks but stays: closing it is next.

**Spelling.** GreynirCorrect changed 43 of 402 correctly written questions: valid words replaced ("sent" ->
"send"), words split ("ennþá" -> "enn þá"), and once a whole second question dropped. `safe_correction()` keeps a
change only where the typed word is unknown to BÍN and the catalog, or where only capitals or Icelandic letters are
restored ("Kopavogi" -> "Kópavogi"), and never drops or adds a word. Clean dev questions changed: 25 -> 4 (capitals
only). Misspelled dev (540): words equal to the original 70.2% typed, 86.8% GreynirCorrect, **88.6% safe**; table
search on it unchanged (47/69/85 -> 46/69/85).

## A search model fine-tuned on JURECA (31 Oct 16:28-16:49, job 15764333, `scripts/finetune_jureca.sbatch`)

One Slurm job on 4 A100s, 21 minutes: vLLM served Qwen3-30B on the node and wrote 4,580 questions (about five per
table, one of them in English) for 924 tables that are in no evaluation set (the 395 tables of dev, test, real, muni, hand, e2e, the new
41 and the organizers' 60 were held out); bge-m3 was trained on those (question, table) pairs for one epoch (144
steps, 30 seconds); the same job then measured the base and the new model on the same sets. Right table first /
top 3 / top 8:

| Set (none of its tables trained on) | bge-m3 | **fine-tuned** |
|---|---|---|
| Dev (270), meaning search alone | 30 / 50 / 65 | **48 / 70 / 86** |
| Dev (270), our full search | 51 / 71 / 86 | **53 / 77 / 90** |
| Municipalities (24) | 54 / 88 / 96 | **67 / 92 / 96** |
| Real questions (11) | 45 / 45 / 64 | 45 / 55 / 64 |
| Hand-written (36) | 56 / 78 / 89 | 47 / 78 / 89 |

The meaning search alone gains 18 points on tables it never saw, and the full search gains 4-6 points in the top
3 and top 8, which is what the picker chooses from. One set lost 3 first places of 36. End to end (run 8) it ties
with plain bge-m3: 56 vs 54 of the 60, 24 vs 25 of the 41. The demo uses plain bge-m3 (calibrated confidence, no
2 GB model to carry); `EMBED_MODEL=<folder>` switches to the fine-tuned one.

## What every answer shows (31 Oct afternoon)

- **The data behind it**, folded: each table (linked), the values read, the **unit from the table's own header**
  (UNITS / UNITS[en]; plain counts left out), the calculation done in code, the response time.
- **What code assumed** for a vague question: "Assuming: Year 2026, Sex Total." / "Miðað við: Ár 2026, Kyn Alls."
- **Relative dates made explicit** before anything else: "í fyrra" -> 2025, "last quarter" -> Q3 2026 (it fires on
  exactly the 10 of 969 dev/test questions that have one).
- **A grey line**: the model, the seconds, whether the numbers were checked, and a **confidence** from signals
  computed in code (do the three searches agree on the table, did the number check pass, did the first table have
  numbers), calibrated with `eval/confidence_eval.py` on questions with known answers.
- **Second opinion**: one click runs the whole pipeline on a second model and says whether both give the same number.

## What each step adds (ablation ladder, `eval/ladder.py`, 31 Oct 01:53)

The same 60 questions, the same model (qwen3-30b), the pipeline built up one step at a time; scored by the
organizers' `evaluate()`. "Numbers" counts answers stating every expected number, from any table or none.

| Rung | Adds | Easy | Medium | Hard | IS | EN | **Right** | Numbers | s/question |
|---|---|---|---|---|---|---|---|---|---|
| 0 no-data | the model alone, no table | 0/20 | 0/20 | 0/20 | 0/30 | 0/30 | **0/60** | 1 | 0.2 |
| 1 embed | bge-m3 search only, first table (close to the starter kit) | 6/20 | 2/20 | 0/20 | 4/30 | 4/30 | **8/60** | 8 | 1.1 |
| 2 checked | our search: fusion + places + checker + spelling | 13/20 | 5/20 | 0/20 | 9/30 | 9/30 | **18/60** | 18 | 0.9 |
| 3 +pick | the model picks among the top 8 | 15/20 | 8/20 | 0/20 | 12/30 | 11/30 | **23/60** | 23 | 0.9 |
| 4 +calc | the model names an operation, code computes | 15/20 | 11/20 | 0/20 | 14/30 | 12/30 | **26/60** | 26 | 0.9 |
| 5 +split | two sub-questions for two tables | 15/20 | 12/20 | 10/20 | 18/30 | 19/30 | **37/60** | 37 | 1.5 |
| 6 +english | answers in the question's language = the full pipeline | 16/20 | 14/20 | 10/20 | 18/30 | 22/30 | **40/60** | 40 | 1.5 |

What it shows:
- **The model alone gets none right.** It recalls one number, but never from a table it could cite: without the
  data the chatbot is worthless, which is the point of the whole design.
- **Our search is the largest single step above the starter's level** (8 → 18).
- **Splitting makes the hard questions possible at all** (0 → 10); calculation in code lifts medium (8 → 11), and
  answering in the question's language lifts English (19 → 22).
- The full pipeline (rung 6) matches run 3 (40/60): the two measurements agree.

Caveats: one question is 1.7 points, so steps of 2–3 questions can be chance; rungs 3–6 include the picker rules
shaped on these 60 questions (see above).

## What we changed, and why

### Before the event
- **Table search** (`hagstofa/retrieve.py`, `checker.py`): three searches fused by rank (character n-grams,
  BÍN lemmas with synonyms, bge-m3 embeddings), then a check that moves tables lacking the question's place or
  year down. Dev set (270 questions): right table first 51% (embeddings alone 30%). Place names ("í Kópavogi" vs
  "Kópavogsbær") and typos handled; measured in `eval/experiments/`.
- **Values, calculations, two tables** (`hagstofa/pipeline.py`): the LLM only chooses (codes, one operation from
  a fixed list); code checks the codes, fetches the numbers and does the arithmetic. "Let code do the maths."
- **Model choice** (`eval/compare_models.py`): given the right table, gemma4:26b chose the right values in 24/24;
  small models failed mostly on the year.

### During the event (30–31 Oct)
1. **Their scoring on our pipeline** (`eval/gagnvist_eval.py`): an adapter, so every number above is measured
   the organizers' way. Only the model's text is scored, never our raw data.
2. **Numbers from the data release** (`hagstofa/local_data.py`): PX-Web blocked us (HTTP 429) for over an hour,
   and the answer key was read from the release (snapshot of 18 Oct). Reading the release's `data.px` files
   reproduces all 100 answer-key cells of the 60 questions; PX-Web is the fallback.
3. **Both editions, with a fallback chain**: Statistics Iceland publishes most tables in Icelandic and English,
   with different codes in ~690 tables and sometimes different numbers. English questions read the English
   edition (English labels, the same numbers as the English answer key); if it fails or is missing, the
   Icelandic one, and the other way round; then the next table by rank; a result must hold at least one number;
   if nothing works, the answer says there is no data. Every fallback taken is recorded with the answer.
4. **English answers**: the answer's language is decided in code, and the instruction is the prompt's last line.
5. **Run 1 → 2 (+8)**:
   - *Forced pick*: the model answered "no table fits" on English questions; now the search's #1 is used.
   - *Calculation across sub-questions*: "Reykjavík and Kópavogur together", "per 1,000" (new operation).
   - *Results kept*: the model re-rounded the code's result ("5,7" for 5,77); any computed result the answer
     lacks is now added, with how it was computed ("Útreikningur: …").
6. **Run 2 → 3** (picker and splitter prompts, `PICK_SYSTEM`, `DECOMPOSE_SYSTEM`, `CALC_SYSTEM`). Unless the
   question asks otherwise, among look-alike tables prefer:
   1. the national total over breakdowns by region, municipality, district, industry or group;
   2. the headline index or series over sub-indices, breakdowns and harmonised (HICP) indices;
   3. current prices over constant prices / real values;
   4. a table that covers the period asked about.

   Also: a per-capita question ("per inhabitant", "per 1,000") is split into the quantity and the population;
   no calculation across parts when the question asks for two separate numbers.
   Misses this targeted in run 2: GDP at constant prices (6 questions), immigrants by municipality (4), private-
   sector wage indices (5), the HICP breakdown (1), per-capita questions kept whole (4).
7. **Server** (`main.py`): the starter's `/spyrja_px_api` contract (their cURL test works on ours) plus every
   table, edition, value chosen, row read, calculation, fallback and the time taken; `/spyrja` answers several
   questions in parallel; the chat UI on the same port.

Still wrong after run 3 (20): look-alike tables remain the main cause: GDP from the summary table THJ01000
instead of THJ01102 (6 questions), the wage index from LAU04001 (indices by group) instead of LAU04000 (5),
immigrants by municipality MAN43005 instead of MAN43000 (3). Some Icelandic questions slipped where their
English twins are right (per-capita, CPI and unemployment): the model's choices are not yet stable across
languages.

## What blocked us
- **Compute**: LUMI accounts were not created in time; JURECA's shared models and container were not ready. We
  ran our own model on JURECA instead: vLLM in our own Python environment, a Slurm job on a GPU node, the
  pipeline on the login node over JURECA's internal network (node name + `i`).
- **vLLM on JURECA**: its FlashInfer sampler compiles GPU code at start-up and the nodes have no `nvcc`;
  `VLLM_USE_FLASHINFER_SAMPLER=0` fixed it.
- **PX-Web rate limit** (429 for over an hour from a shared Starlink address): solved by the data release.
- **SSH from Windows**: "Corrupted MAC on input" (fixed with `Ciphers aes256-gcm@openssh.com`); idle
  connections dropped (fixed with `ServerAliveInterval`).

## Known limits
- Look-alike tables remain the main source of wrong answers; the rules help but are prompts, not guarantees.
- Questions Statistics Iceland can't answer are mostly answered from some table (declined 3 of 16).
- New questions score well below the 60 our rules were shaped on (see the 41).
- The release is a snapshot of 18 Oct 2026: newer figures come only from PX-Web.

## What we would do next
- Decline what is not published, without refusing real questions (a "nodata" check refused 2026 as "the future").
- Fine-tune the search model again with more English questions, and with real questions from people.
- Measure on new, real questions from people, not the 60 the rules were shaped on.
