# The Self-Auditing Ledger — Project Overview & Finalized Architecture

*A complete, plain-language explanation of what we are building, why, how it
works, what we have done so far, and what comes next. Written so it can be
presented and defended to a reviewer. This document also serves as the **locked
architecture** — once approved, we build from exactly what is described here.*

---

## How to read this document

It is long because it is complete, but it is built in layers. If you only have
five minutes, read **Section 0 (Executive Summary)** and **Section 14 (The Locked
Decisions)**. Everything in between explains *why* those decisions are correct.
Section 15 is a plain-language glossary — if any word confuses you, look there.

A note on the word "phase":
- **Phase 1 / Phase 2** = the two *conceptual* halves of the system (build the
  graph and extract features → train a model).
- **Build Step 0–6** = our *work plan* (the order we actually do things).

These are different things. We keep them separate on purpose.

---

## 0. Executive Summary (the elevator pitch)

**What it is.** An ERP* fraud-detection research system. Instead of judging each
transaction on its own, we turn a company's transaction history into a
**temporal graph** (a network of who-paid-whom, stamped with time) and look for
**suspicious shapes** in that network — money moving in circles, many accounts
funneling into one, rapid multi-step transfers. We then feed those "shape"
measurements to a machine-learning model that scores each transaction's fraud
risk, and we explain every alert by showing the exact suspicious path.

**The honest scientific claim.** A real-time production platform is the long-term
*vision*. The thing we actually set out to *prove* is narrower and defensible:

> **Do graph-shape ("topology") features add real predictive value on top of a
> strong ordinary-transaction baseline — when measured fairly, with no cheating
> (no data leakage)?**

**Why this is worth doing.** A previous version of this project claimed "graph
topology is the differentiator," but when we audited it, its *own results
contradicted that claim*, and all its scores were inflated by data leakage (the
model was accidentally allowed to see the future). So the central idea has never
actually been tested fairly. We rebuilt the project to test it properly — and we
commit to reporting the result **either way**, even if it turns out topology does
*not* help. Designing the fair test is the contribution.

**Key talking points for the presentation:**
1. We are testing a hypothesis, not just shipping a tool.
2. We found and fixed a serious flaw (data leakage) in the prior approach.
3. Our architecture is designed so that "cheating" (seeing the future) is
   structurally impossible.
4. We pick datasets along a spectrum of "graph richness" so we can show *when*
   graph structure helps and when it doesn't.
5. We will report a negative result honestly if that is what the data shows.

*ERP = Enterprise Resource Planning, the software (SAP, Oracle, etc.) that records
a company's financial transactions. See glossary.*

---

## 1. The big idea, in plain English

Most fraud-detection systems look at one transaction at a time: *Is this $5,000
payment unusual on its own?* That misses an entire class of fraud, because real
financial fraud is usually **a pattern made of several connected transactions**,
not one obviously-bad payment. Examples:

- **Circular payments:** A pays B, B pays C, C pays A. Money goes in a loop.
  Legitimate business almost never sends money in circles.
- **Lapping:** new incoming money is used to cover a previously stolen amount,
  forming a chain of "robbing Peter to pay Paul."
- **Fan-in / collection:** many accounts rapidly funnel money into one account
  (a classic money-mule collection pattern).
- **Rapid transfer chains:** money hops A→B→C→D→E in seconds (layering, to hide
  the trail).

None of these look wrong if you stare at a single row in a spreadsheet. They only
become visible when you **draw the money flow as a network and look at its
shape**. That network is what we call the **Shadow Graph** — a live "shadow" or
mirror of the company's real transactions:

```
   account = a dot (node)        payment = an arrow (edge), stamped with time + amount

        (A) ──$10k, 10:00──▶ (B) ──$10k, 10:03──▶ (C)
         ▲                                          │
         └──────────────$10k, 10:05────────────────┘      ←  a 3-step money LOOP
```

The word **temporal** matters: every arrow carries a timestamp, so the graph
preserves both **structure** (who connects to whom) and **chronology** (in what
order). Fraud patterns are defined by *both* — a loop that closes within 5 minutes
is suspicious; the same accounts transacting months apart is not.

---

## 2. The scientific question we are actually answering

It is important (especially for a research project) to separate the **vision**
from the **proof**.

- **The vision** (from the original project description): a real-time platform
  that watches an ERP system live, maintains the Shadow Graph, and raises instant,
  explainable fraud alerts. This is the inspiring story and the demo we will build.
- **The proof** (what earns a defensible result): a careful, leakage-free
  experiment answering one question —

> **Claim:** Topology features (the graph-shape measurements) improve fraud
> detection compared to a strong baseline that uses only ordinary transaction
> features — and this improvement survives a fair, no-leakage evaluation.

Everything in the architecture is built to make that claim **testable and
trustworthy**. The decisive test is described in Section 9 (the "ablation").

### Is this still a real-time auditor? Yes.

"Vision vs. proof" is about what we scientifically *claim*, **not** about what we
build. We are **not** dropping the real-time, explainable auditor — we are building
it. The distinction only means: we *prove* the narrow, measurable claim above
(rather than claiming "we shipped a finished product").

The reassuring part: **our leakage-free design and the real-time design are the
same design.** The core rule — "for each transaction, look only at the past, then
move on" — is *literally how a real-time system works* (it reacts to events as they
arrive and cannot see the future). So being scientifically careful did not cost us
the real-time auditor; it **is** the real-time auditor. We get rigor for free.

What that means concretely:

| We ARE building | We are NOT claiming |
|---|---|
| The streaming engine: event in → graph updates → topology features from the past → risk score | A product deployed live inside a company's running SAP server |
| The explainable audit alert (SHAP + the detected fraud path, shown in Neo4j) | Millisecond-latency production hardening / live ERP integration |
| A **live demo**: replay a real dataset transaction-by-transaction through the engine and watch alerts fire | "We shipped a finished commercial platform" |

The demo runs on **recorded** datasets replayed *as if* streaming — which is
necessary for research, because we need *labeled* data (known fraud / not-fraud) to
measure accuracy; a live production system never tells you the ground truth.

**One-line framing for the presentation:** *"We built the real-time, explainable
fraud-detection engine, **and** we rigorously proved its graph features add value
without leakage."* Two deliverables, both real — not one instead of the other.

---

## 3. Why we rebuilt — what went wrong before (and why that is good to present)

There was an earlier version of this project (kept in the repo under `reference/`,
read-only). It ran 6 datasets and concluded *"graph topology is the ultimate
differentiator."* We audited it thoroughly before writing any new code. We found
two serious problems. Presenting this honestly is a strength — it shows scientific
rigor.

**Problem 1 — the headline claim was contradicted by its own results.**
On its flagship dataset, the model's own "feature importance" report showed the
predictive signal came almost entirely from `total_amount` (62%) and a couple of
amount statistics. **Every graph-shape feature — cycles, lapping, density, chains —
scored 0.0 importance.** The very features the project was about contributed
nothing. The success was claimed, not demonstrated.

**Problem 2 — the impressive scores were inflated by data leakage.**
"Data leakage" means the model was accidentally allowed to use information it would
not have at prediction time — usually information from the **future**. This makes
test scores look great but they collapse in the real world. There were two leaks
stacked together:

- *The feature engine looked into the future.* It computed each transaction's
  graph features using the **entire** graph, including transactions that happened
  *later*. One of its top features was literally "time until the **next**
  transaction." That is using tomorrow to predict today.
- *The evaluation protocol leaked too.* It mixed past and future randomly when
  splitting data for testing, used a careless way of handling the rare-fraud
  imbalance, and never tuned its decision cutoff. (Details in Section 8.)

**Conclusion.** The good idea was never given a fair test. The old numbers (e.g.
"99.7% AUC") are not trustworthy. So we are not "improving" those numbers — we are
**discarding them and producing the first honest ones.** We keep the old code only
as a source of proven data-cleaning tricks, never its model/evaluation code.

---

## 4. The finalized architecture (LOCKED)

This is the pipeline. Data flows top to bottom. Each box is one responsibility.

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  RAW DATA  (transaction CSVs from public fraud datasets)          │
   └───────────────────────────────┬──────────────────────────────────┘
                                    ▼
   (1) ETL  ─ "clean + standardize" ───────────────────────────────────
        Turn each dataset's messy raw columns into ONE canonical form:
        Accounts (nodes) + Transactions (edges with time, amount, label).
                                    ▼
   (2) SHADOW GRAPH  ─ the temporal network of money flow ─────────────
        Accounts = nodes;  transactions = time-stamped directed edges.
                                    ▼
   (3) TEMPORAL TOPOLOGY ENGINE  ─ "find the shapes" ──────────────────
        For each transaction, measured ONLY from the past (no future!):
        cycle? fan-in? rapid chain? lapping? how dense/central?  → numbers
                                    ▼
   (4) FEATURE TABLE  ─ one row per transaction ──────────────────────
        ordinary features (amount, time gaps, account info)
        + topology features (the shape measurements)  + the fraud label
                                    ▼
   (5) MODEL  ─ machine learning risk scorer ─────────────────────────
        Gradient-boosted trees (XGBoost / LightGBM), trained the RIGHT way
        (time-aware, one imbalance fix, tuned cutoff)  → fraud probability
                                    ▼
   (6) EVALUATION + THE ABLATION  ─ "did topology actually help?" ─────
        Compare: baseline (ordinary only)  vs  ordinary + topology.
        Measured with PR-AUC and precision@k on a future test period.
                                    ▼
   (7) EXPLAINABLE AUDIT ALERT  ─ "why was this flagged?" ─────────────
        SHAP (which features drove the score) + the actual suspicious
        path drawn from the graph (shown live in Neo4j for the demo).
```

### The one big engineering decision (and why)

We compute everything in **Python (Pandas + NetworkX/igraph)**, and use **Neo4j
only for the visual demo and path explanations** — *not* for the heavy
computation. Why: the old version tried to do the heavy computation inside Neo4j
and it **ran out of memory** on the larger datasets, which forced it to silently
turn topology off (another reason its results were meaningless). Python computes
the features reproducibly, at scale, and lets us guarantee no leakage. Neo4j is
where we *show* the shadow graph and the detected fraud path — its real strength
(visualization and explanation), without betting the whole pipeline on it.

### The code layout (so the work maps cleanly to the architecture)

| Folder (`src/…`) | Architecture box | Responsibility |
|---|---|---|
| `common`   | (all)        | Config loader, logging, Neo4j connection |
| `etl`      | (1)          | Raw CSV → canonical accounts + transactions |
| `graph`    | (2)          | Load the graph into Neo4j (for the demo) |
| `topology` | (3)          | The detectors + structural features (single copy) |
| `features` | (4)          | Assemble the per-transaction feature table |
| `modeling` | (5)(6)       | Time-aware training, cutoff tuning, evaluation, saving |
| `experiments` | (6)       | `run_dataset.py` (full pipeline), `ablation.py` (the test) |

---

## 5. The datasets — and why we use several of them

A subtle but important design choice. Our question is *"does graph structure
help?"* — so the natural thing to vary across datasets is **how much real graph
structure each one has.** We line the datasets up on a spectrum from "rich
structure" to "no structure," and run the **same test** on each. If topology
features help more on the structure-rich datasets and less on the flat ones, that
pattern is strong evidence. (The old project compared *across* datasets that
differed in *everything* at once, which proves nothing. We instead test *within*
each dataset, then look at the trend.)

| Tier | Dataset | Size | Fraud rate | Graph structure | Role in our study |
|---|---|---|---|---|---|
| **1** | **IBM AML** | 1.32M txns | 0.13% | **Rich** — real account↔account, true cycles | The decisive testbed for topology |
| **1** | **BankSim** | 595k txns | 1.21% | **Sparse** — customer→merchant only, no cycles possible | The control: topology *should* help less here |
| 2 | **SAP Würzburg** | 59,852 docs | 0.08% (49 frauds) | Thin ERP "star" | Real ERP domain + the explainability demo |
| 2 | **PaySim** | 6.36M txns | 0.13% | Rich, at large scale | Proves the engine scales; transfer→cash-out fraud |
| 3 | **FIFAR** | 506k rows | 1.13% | None (pure tabular) | Optional "floor": no graph at all |
| — | ~~SAP IDES~~ | — | — | **Dropped** | A trial-watermarked file left only 7 frauds — unusable |

**Why this is the right answer to "should we use all of them?"** Yes — but for a
*reason*, not just to show work. Each dataset is a different point on the
structure spectrum, so together they map out *when* the method helps. We do them
in **tiers** (start with the two that matter most) so we get a trustworthy answer
quickly before investing in the giant 6.3M-row dataset.

**About fraud rates:** notice every dataset is **extremely imbalanced** — fraud is
0.08%–1.2% of transactions. This single fact drives almost all of our modeling
rules (Section 8). When 99.9% of cases are "not fraud," ordinary accuracy is
worthless (a model that says "never fraud" is 99.9% accurate and catches nothing).

---

## 6. Phase 1, part A — building the Shadow Graph (the data layer)

**Goal:** turn five very different raw datasets into one consistent shape so the
rest of the pipeline does not care which dataset it is looking at.

**The canonical form** (the single standard we convert everything into):
- **Nodes = Accounts.** Each account is one dot, with attributes when available
  (country, account type, starting balance, "known bad actor" flag).
- **Edges = Transactions.** Each transaction is one arrow from sender account to
  receiver account, carrying `amount`, `timestamp`, and the fraud `label` (0/1).
- **The prediction unit is one transaction** (one edge). We score each transaction.

**The single most important fix vs. the old version — the "unified account
namespace":** In the datasets with real account-to-account money flow, the old ETL
put senders in one bucket ("Account") and receivers in a *different* bucket
("Vendor"). That meant account #6456-as-a-sender and #6456-as-a-receiver became
**two different dots** — so a loop A→B→A was *impossible to see*. We verified this
was the bug and fixed it: **senders and receivers share one account identity**, so
loops and chains are visible again. (We confirmed in the data that 99.3% of sender
IDs also appear as receiver IDs — they really are the same accounts.)

---

## 7. Phase 1, part B — the Temporal Topology Engine (the heart of the project)

This is where we turn **shapes into numbers**. It is also the part most vulnerable
to leakage, so its design is the most important thing in the whole project.

### 7.1 The golden rule: only ever look at the past (no look-ahead)

**The auditor analogy.** Imagine an auditor sitting at a desk as transactions
arrive one by one. When transaction #500 lands, the auditor may look at #1–#500 to
judge it — but *cannot* see #501 onward, because those have not happened yet. Our
engine works **exactly** like this:

> Process transactions in time order. For a transaction at time **T**, compute its
> features using only transactions with timestamp **≤ T** (optionally only those in
> a recent window, e.g. the last 24 hours or 7 days). Then add it to the graph and
> move on.

Because we *only ever look at edges already added*, **the future cannot leak in —
it is structurally impossible.** This is the same "incremental / streaming" idea
the original vision describes for real-time operation, so the realistic design and
the leakage-safe design are the **same** design. Two supporting rules:

1. **No whole-dataset statistics.** Any average or z-score (e.g. "is this amount
   unusual for this account?") is computed only over that account's *past*
   transactions, growing as time goes on — never over the entire dataset (which
   would include the future).
2. **Credit the closer, not the openers.** When a loop A→B→C→A finally closes, the
   "this is part of a cycle" signal is attached to the transaction that *closed*
   it (the one we are scoring now) — not pasted backward onto A's earlier payment
   (which, at the time, could not have known a loop would form).

We will also include an automated **leakage test**: scramble the *future*
transactions and confirm a past transaction's features do **not** change. If they
change, we have a leak and the build fails.

### 7.2 The detectors (what shapes we look for)

Each detector scans the *as-of-T* graph (past only, within a time window) and
outputs a few numbers per transaction. We start with the two that match the
**known** fraud shapes in our decisive dataset (IBM AML's frauds are literally
labeled as cycles and fan-ins), then add the rest:

| Detector | The shape it finds | Plain meaning |
|---|---|---|
| **Cycle** | A→B→…→A loop within a time window | Money returning to its origin — laundering / round-tripping |
| **Fan-in / Fan-out (density)** | Many edges into/out of one account fast | Collection (mule) or distribution hub |
| **Rapid chain** | A→B→C→D… with tiny time gaps | Layering — moving money fast to hide the trail |
| **Lapping** | Near-equal sequential amounts on one entity | Covering a prior shortfall with new money |
| **Real centrality** | PageRank / betweenness / degree on the as-of graph | How "central" or busy an account is in the network |

Each becomes features like: `in_cycle` (yes/no), `cycle_length`, `cycle_amount`,
`fan_in_count`, `max_chain_length`, `account_pagerank`, etc. (The old version's
"centrality" was fake — it was just counting edges. Ours uses real graph
algorithms.)

### 7.3 The output

A **feature table**: one row per transaction, columns = ordinary features +
topology features + the label. This table is what Phase 2 trains on. Crucially,
the ordinary and topology columns are kept clearly separable so we can switch
topology on and off for the ablation (Section 9).

---

## 8. Phase 2 — how the model works (and the rules that keep it honest)

### 8.1 The two-phase logic

The graph engine **finds and measures** the shapes (Phase 1). The model does *not*
re-discover shapes — it **learns which shape-measurements actually predict real
fraud** and combines them with ordinary features into a single risk score
(Phase 2). Input = the feature row; output = a fraud probability (0 to 1).

### 8.2 Which model, and why

We use **gradient-boosted decision trees** (XGBoost / LightGBM) as the workhorse,
with logistic regression as a simple baseline. Why gradient-boosted trees:
- They are the state of the art on **tabular** (table-shaped) data like ours.
- They handle **imbalance** well.
- They are **interpretable** — we can see exactly which features drove a score
  (via SHAP). That matters because **explainability is our selling point.**

We deliberately treat fancy Graph Neural Networks (GNNs) as *optional future
comparison only* — they are black boxes, which works against our goal of giving
auditors clear, explainable reasons.

### 8.3 The five non-negotiable rules (each is a fix for a real old-version flaw)

These are the difference between trustworthy and meaningless results. Each is
simple once explained:

1. **One imbalance fix, never two.** Because fraud is so rare, models need help
   noticing it. There are two common tools: *resampling* (e.g. SMOTE, which
   invents synthetic fraud examples) and *class weights* (telling the model "fraud
   mistakes count more"). The old version used **both at once**, which
   double-counts and distorts. We use **exactly one — class weights** — per
   experiment. (We avoid SMOTE for graph features because inventing a synthetic
   "halfway between two cycles" example is physically meaningless.)

2. **Tune the decision cutoff; never assume 0.5.** A model outputs a probability;
   we need a cutoff to call it "fraud." The default 0.5 is arbitrary. We **choose**
   the cutoff on a validation period (e.g. the one that best balances catching
   fraud vs false alarms), then **freeze** it before touching the test data.

3. **Time-aware testing; never shuffle time.** We train on **earlier** transactions
   and test on **later** ones — exactly how the real world works (you predict the
   future from the past). The old version shuffled transactions randomly, letting
   the model effectively "study the answers" from the future. That is the single
   biggest evaluation leak, and we forbid it.

4. **No look-ahead through the graph.** (Already enforced in Phase 1, Section 7.1 —
   listed again here because it is a modeling-integrity rule too.)

5. **Right metrics for rare events.** We report **PR-AUC** (a.k.a. average
   precision) — which focuses on how well we find the rare frauds — as the *primary*
   metric, and **precision@k / recall@k** ("of the top-N alerts an auditor actually
   reviews per day, how many are real fraud?"). We de-emphasize plain accuracy and
   ROC-AUC, which look flattering on imbalanced data. We never pick the "best"
   model by its 0.5-cutoff score.

**One more rule — keep the evidence.** Every trained model is saved together with
its data-scaler and its frozen cutoff, so any result can be reproduced exactly.

---

## 9. The experiment that decides everything — the ablation

This is the core scientific test, and the old project never ran it.

> Take one dataset. Use the **same** time-aware split, the **same** model, the
> **same** single imbalance fix. Train it **twice**:
> - **(A) Baseline:** ordinary transaction features only.
> - **(B) Baseline + Topology:** the same, plus our graph-shape features.
>
> Compare their PR-AUC (and precision@k). Repeat for each dataset.

- If **(B) clearly beats (A)** — especially on the structure-rich IBM AML data —
  then the thesis is **proven**, properly and for the first time.
- If **(B) does not beat (A)**, that is an honest, publishable **negative result**:
  *"explicit topology features add no value beyond ordinary features on these
  datasets."* That is far more credible than the old, leakage-inflated "success,"
  and still a real scientific contribution (a fair test designed and run).

We commit, up front, to reporting **whichever way it lands.** (`experiments/
ablation.py` is the script that runs this.)

---

## 10. Explainability — the audit alert (our differentiator)

A risk score of "0.97" is useless to an auditor who must justify an investigation.
So when a transaction is flagged, we produce a human-readable alert with two parts:

1. **Why the model scored it high** — via **SHAP**, which lists the features that
   pushed the score up (e.g. "+0.4 because it closes a 3-step cycle within 5
   minutes; +0.2 because the amount is unusual for this account").
2. **The actual evidence** — the concrete suspicious **path** drawn from the
   graph (e.g. `Vendor_A → Vendor_B → Vendor_C → Vendor_A, $50,000, 5 minutes`),
   shown visually in **Neo4j** for the demo.

This is the part of the original vision most worth keeping, and the old version
never actually built it.

---

## 11. What we have done so far (concrete progress)

**Build Step 0 — Audit & honest re-baseline. ✅ Done.**
- Read the entire old project (feature engine, ETL, all training scripts, reports).
- Confirmed the two leakage problems and the "topology scored 0 importance"
  finding (Section 3). This is *why* we rebuild.

**Architecture & decisions — Locked. ✅ Done.**
- All major decisions agreed and written into `CLAUDE.md` (the project rulebook),
  `docs/DESIGN.md` (the technical reasoning), and this document.

**Build Step 1 — ETL / Shadow Graph for the Tier-1 datasets. ✅ Done & verified.**
We built the canonical-graph builder and ran it on both Tier-1 datasets:

| Dataset | Transactions | Accounts | Fraud rate (verified) | Namespace overlap | Result |
|---|---|---|---|---|---|
| **IBM AML** | 1,323,234 | 10,000 | 0.13% (1,719 frauds) | **99.3% → unified ✓** | Cycles now detectable |
| **BankSim** | 594,643 | 4,162 | 1.21% (7,200 frauds) | **0.0% → bipartite ✓** | Confirmed as the sparse control |

- The unified-account fix is **working and verified** (it was the #1 bug before).
- **A key finding that makes IBM AML the perfect testbed:** *every one* of its
  1,719 frauds is a known shape — **936 cycles + 783 fan-ins**. So if our graph
  features cannot beat the baseline here, they cannot anywhere — and we have the
  ground-truth shape labels to check our detectors against.
- We also discovered two bonus files the old version ignored: per-account
  attributes (`accounts.csv`) and the fraud-shape labels (`alerts.csv`). We use the
  shape labels for *evaluation only* — never as a model input (that would be
  cheating).

**In short: the foundation (clean, correct, leakage-aware data layer) is built and
checked for the two most important datasets.**

---

## 12. What comes next (the roadmap)

| Build Step | What | Status |
|---|---|---|
| 0 | Audit & re-baseline | ✅ Done |
| 1 | ETL / Shadow Graph (Tier-1) | ✅ Done |
| **2** | **Temporal Topology Engine** — the leakage-free feature extractor (Section 7) | ◀ **Next** |
| 3 | Modeling harness — time-aware training, cutoff tuning, metrics, saving (Section 8) | Planned |
| 4 | **The Ablation** — does topology help? (Section 9) | Planned |
| 5 | Explainability — SHAP + Neo4j audit-alert demo (Section 10) | Planned |
| 6 | Optional — PaySim scale, GNN comparison, synthetic ERP generator | Later |

The immediate next step (Build Step 2) is the algorithmic core: the as-of /
streaming feature engine with the cycle and fan-in detectors, plus the automated
leakage test.

---

## 13. The technology stack (one place)

- **Python** — everything.
- **Pandas / NumPy** — data handling.
- **NetworkX / igraph** — graph computation (the compute path).
- **Neo4j** — graph visualization & path explanation (the demo path only).
- **scikit-learn, XGBoost, LightGBM** — the models and evaluation.
- **SHAP** — model explanations.
- **Matplotlib / Seaborn / Plotly** — plots and interactive visuals (Plotly for
  interactive result charts + standalone interactive fraud-path figures).

---

## 14. The LOCKED decisions (the architecture we commit to)

After approval, we build from exactly this. Nothing here changes without a
deliberate, recorded decision.

1. **Two-phase design:** (Phase 1) build a temporal Shadow Graph and extract
   topology features per transaction; (Phase 2) train an ML model on the resulting
   feature table.
2. **Goal = both** a rigorous research result *and* an explainable demo.
3. **Compute in Python (Pandas + NetworkX/igraph); Neo4j is the demo/explanation
   layer only.**
4. **Canonical data model:** Accounts (nodes) + Transactions (time-stamped edges);
   senders and receivers share **one** account namespace; the prediction unit is
   one transaction.
5. **Leakage-free by construction:** as-of / streaming feature computation, past
   only; no whole-dataset statistics; signal credited to the transaction that
   completes a structure; an automated leakage test guards it.
6. **Detectors:** cycle, fan-in/fan-out (density), rapid chain, lapping, plus real
   centrality. (Cycle + fan-in first, matching IBM AML's known fraud shapes.)
7. **Model:** gradient-boosted trees (XGBoost/LightGBM) + logistic baseline;
   GNNs are optional future comparison only.
8. **Honest evaluation:** time-aware splits; exactly one imbalance mechanism
   (class weights by default); decision cutoff tuned on validation and frozen;
   PR-AUC primary + precision@k; models saved with scaler and cutoff.
9. **Validation method:** the within-dataset **ablation** (baseline vs
   baseline+topology) is the decisive test; we report the result either way.
10. **Datasets (tiered):** Tier 1 IBM AML + BankSim; Tier 2 SAP Würzburg + PaySim;
    Tier 3 FIFAR (optional). SAP IDES dropped.
11. **Explainability:** SHAP feature attributions + the detected graph path, shown
    in Neo4j.
12. **Synthetic ERP generator:** deferred (and, if built later, designed to avoid
    the circularity trap — fraud injected by behavior, not by detector pattern).

---

## 15. Glossary (plain-language)

- **ERP:** the software that records a company's transactions (SAP, Oracle, etc.).
- **Topology / graph structure:** the *shape* of the money-flow network — who
  connects to whom, in what pattern.
- **Temporal graph:** a network where every connection has a timestamp, so order
  and timing matter, not just structure.
- **Shadow Graph:** our live network mirror of the ERP's transactions.
- **Node / Edge:** a node is a dot (an account); an edge is an arrow (a transaction).
- **Data leakage:** accidentally letting the model use information it would not
  have at prediction time (usually from the future). Makes test scores look great
  but they fail in reality.
- **Look-ahead:** a specific leak — using future transactions to compute a present
  transaction's features.
- **As-of / streaming computation:** computing each transaction's features using
  only data available *as of* its own time — the cure for look-ahead.
- **Class imbalance:** when one class (fraud) is extremely rare (here <1.3%).
- **SMOTE:** a method that invents synthetic minority (fraud) examples to balance
  the data. We avoid it for graph features (synthetic "half-cycles" are meaningless).
- **Class weights:** telling the model that mistakes on the rare class count more.
  Our chosen imbalance fix.
- **Decision threshold / cutoff:** the probability above which we call something
  fraud. We tune it instead of assuming 0.5.
- **Cross-validation:** a way of testing a model on data it did not train on. We use
  a **time-aware** version (train on the past, test on the future).
- **PR-AUC / Average Precision:** a metric that measures how well we find the rare
  positives (fraud). Our primary metric on imbalanced data.
- **Precision@k / Recall@k:** of the top-k alerts an auditor actually reviews, how
  many are real fraud (precision), and what fraction of all fraud do they cover
  (recall). The operationally honest metric.
- **ROC-AUC:** a popular ranking metric that looks flatteringly high on imbalanced
  data — we report it but do not rely on it.
- **Ablation:** an experiment that removes one ingredient (here, the topology
  features) to measure how much it was contributing.
- **Cycle / Fan-in / Lapping / Chain:** the fraud shapes (Section 1 / 7.2).
- **Segregation of Duties (SoD):** an internal-control rule that one person should
  not control multiple steps (create + approve + pay) of a transaction. Only the
  SAP dataset could support this detector, so we treat it as ERP-specific.
- **SHAP:** a method that explains a model's score by how much each feature pushed
  it up or down.
- **GNN (Graph Neural Network):** an ML model that learns directly on graphs. Powerful
  but a black box — we keep it as optional future comparison, not the core.
- **XGBoost / LightGBM:** gradient-boosted decision-tree models; our workhorse.
