# Where the strong reference model "fails" (gpt-oss-120b via Groq, few-shot, first 100 test examples)

gpt-oss-120b scored 92/100 exact match. Reading its 8 misses one by one shows that **7 of the 8 are label
conventions of the synthetic dataset, not model errors**. Only one is a real disagreement about behaviour.

| # | Request (abridged) | Gold label | gpt-oss-120b | Verdict |
|---|---|---|---|---|
| 1 | Chinese restaurant in New York | adds `"name": ""` | omits empty `name` | dataset artefact (empty-string filler) |
| 2 | Directions New York → Los Angeles | adds `"mode": "driving"` | omits `mode` | dataset artefact (invented default) |
| 3 | Meeting tomorrow 10–11 am | call with `2022-04-15T10:00:00` | declines (no date known) | defensible either way; the gold date is invented |
| 4 | Sci-fi book published in 1984 | `"keyword": ""` | `"keyword": "science fiction"` | dataset artefact |
| 5 | Upcoming events in New York next week | invents `"date": "2022-05-15"` | omits date | dataset artefact (hallucinated date) |
| 6 | Birthday party on the 25th next month | 2022 dates, title "My Birthday Party" | 2026 dates, "Birthday Party" | dataset artefact (glaive's frozen "today" is 2022) |
| 7 | News on the *recent* climate conference | `"climate change conference"` | keeps "recent" in the keywords | paraphrase, arguably both fine |
| 8 | Hotel in New York 15–20 June | `2022-06-15` | `2026-06-15` | dataset artefact (year) |

**Takeaway used in the README:** exact match on glaive measures *agreement with the dataset's conventions*
(empty-string fillers, invented defaults, the year 2022) as much as raw capability. Fine-tuning learns those
conventions; a stronger prompted model does not. That is precisely why a team fine-tunes for a fixed schema —
and also why a production version should normalise or drop such labels before training (see Future work).
