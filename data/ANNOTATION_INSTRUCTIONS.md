# Ground Truth Annotation Instructions

## What You Need to Do

You need to create **48 ground truth entries** (8 contracts × 6 queries each) by reading actual CUAD contracts and writing correct answers. This is the **only manual step** — everything else (running tests, computing metrics, statistics) is automated.

**Time estimate:** ~4-6 hours total (30-45 min per contract)

---

## Step-by-Step Process

### Step 1: Pick 8 Contracts

Open the CUAD folder: `C:\Users\suyas\Downloads\archive (1)\CUAD_v1\full_contract_txt`

**Recommended contracts** (these have the most cross-references and defined terms, which makes them ideal for testing all 3 trap types):

| # | Contract File | Why It's Good |
|---|---|---|
| 1 | `BERKELEYLIGHTS,INC_06_26_2020-EX-10.12-COLLABORATION AGREEMENT.txt` | 47 cross-refs, 77 defined terms |
| 2 | `Array BioPharma Inc. - LICENSE, DEVELOPMENT AND COMMERCIALIZATION AGREEMENT.txt` | 34 cross-refs, 70 defined terms |
| 3 | `AimmuneTherapeuticsInc_20200205_8-K_EX-10.3_11967170_EX-10.3_Development Agreement.txt` | 25 cross-refs, 72 defined terms |
| 4 | `BELLICUMPHARMACEUTICALS,INC_05_07_2019-EX-10.1-Supply Agreement.txt` | 26 cross-refs, 41 defined terms |
| 5 | `Antares Pharma, Inc. - Manufacturing Agreement.txt` | 13 cross-refs, 28 defined terms |
| 6 | `AtnInternationalInc_20191108_10-Q_EX-10.1_11878541_EX-10.1_Maintenance Agreement.txt` | 19 cross-refs, 40 defined terms |
| 7 | `BONTONSTORESINC_04_20_2018-EX-99.3-AGENCY AGREEMENT.txt` | 16 cross-refs, diverse sections |
| 8 | `BIOCEPTINC_08_19_2013-EX-10-COLLABORATION AGREEMENT.txt` | 8 cross-refs, 14 defined terms |

### Step 2: For Each Contract, Create 6 Queries

You need **2 queries per trap type**:

#### Trap A: "Invisible Exception" (2 queries per contract)

**What to look for:** A clause that has an exception or override elsewhere in the contract.

**How to find them:**
1. Open the contract in a text editor
2. Search for these keywords: `notwithstanding`, `except as`, `subject to`, `provided however`, `unless otherwise`
3. When you find one (e.g., "Notwithstanding Section 7.2, the Company shall not..."), you've found your test case

**How to write the query:**
- Write a question about the **main clause** (Section 7.2 in the example), NOT the exception
- A naive RAG system will only retrieve the main clause and miss the exception
- Example: "What are the Company's indemnification obligations under this agreement?"   

**How to write the ground truth answer:**
- Write the COMPLETE answer that includes BOTH the main clause AND the exception
- Include section references
- Example: "Under Section 7.2, the Company shall indemnify Licensee for all third-party IP claims. However, per Section 9.5, notwithstanding Section 7.2, this indemnification is capped at $5M and does not apply to claims arising from Licensee's modifications."

#### Trap B: "Distant Definition" (2 queries per contract)

**What to look for:** A defined term used in a clause far from where it's defined.

**How to find them:**
1. Search for Article 1 / Section 1 — this is usually the "Definitions" section
2. Look for terms in quotes with "means" or "shall mean" — e.g., `"Change of Control" means...`
3. Then search for where that term is USED later in the contract (usually 20+ pages away)
4. The usage clause + the definition clause = your test case

**How to write the query:**
- Ask about the clause that USES the defined term, without defining it yourself
- Example: "What restrictions apply in the event of a Change of Control?"

**How to write the ground truth answer:**
- Include BOTH the restriction clause AND the definition of the term
- Example: "Section 12.3 states that in the event of a Change of Control, the non-affected party may terminate within 30 days. 'Change of Control' is defined in Section 1.5 as: (a) any merger where shareholders hold less than 50% of voting power, (b) sale of substantially all assets, or (c) acquisition of 50% or more of outstanding shares."

#### Trap C: "Scattered Components" (2 queries per contract)

**What to look for:** Information spread across 3+ different sections.

**How to find them:**
1. Think about cumulative questions: "List ALL termination triggers" or "What are ALL the obligations of Party A?"
2. These answers naturally span multiple sections

**How to write the query:**
- Ask a question that requires gathering info from multiple places
- Example: "List all circumstances under which either party may terminate this agreement."

**How to write the ground truth answer:**
- Collect ALL relevant pieces from across the contract
- Number them and include section references
- Example: "Termination can occur under the following circumstances: (1) Material breach with 30-day cure period (Section 8.1(a)), (2) Insolvency or bankruptcy (Section 8.1(b)), (3) Failure to achieve milestones per Schedule A (Section 8.2), (4) Convenience with 90-day notice (Section 8.3), (5) Force majeure exceeding 180 days (Section 15.4)."

### Step 3: Write Information Units

For each query, break the answer into atomic "information units" — discrete facts that must be present for the answer to be complete. This is the **checklist** that measures how thorough each system's answer is.

**Example for a termination query:**
```
"information_units": [
    "Material breach is a termination trigger",
    "30-day cure period required for breach",
    "Insolvency or bankruptcy is a termination trigger", 
    "Failure to achieve milestones per Schedule A triggers termination",
    "Termination for convenience requires 90-day notice",
    "Force majeure exceeding 180 days allows termination"
]
```

**Rule of thumb:** Each query should have 3-8 information units.

### Step 4: Fill in `data/ground_truth.json`

Open `data/ground_truth.json` and replace the template entry with your actual annotations.

**Entry format:**
```json
{
    "id": "berkeley_trap_a_01",
    "contract_file": "BERKELEYLIGHTS,INC_06_26_2020-EX-10.12-COLLABORATION AGREEMENT.txt",
    "query": "Your actual query here",
    "trap_type": "trap_a",
    "difficulty": "multi_hop",
    "ground_truth_answer": "Your complete correct answer here with section references",
    "relevant_sections": ["Section 7.2", "Section 9.5"],
    "information_units": [
        "Company indemnifies for third-party IP claims",
        "Indemnification is capped at $5M per Section 9.5",
        "Does not apply to claims from Licensee modifications"
    ],
    "notes": "Section 9.5 uses 'notwithstanding' to override Section 7.2"
}
```

**ID naming convention:** `{contract_shortname}_{trap_type}_{number}`
- Example: `berkeley_trap_a_01`, `berkeley_trap_a_02`, `berkeley_trap_b_01`, etc.

### Step 5: Validate Your Annotations

After filling in all 48 entries, run:

```
python -m data.ground_truth_loader
```

This will check your file for:
- Missing/placeholder entries
- Entries with very short answers
- Missing information units
- Distribution across trap types and contracts

You should see: **48 valid entries, 16 per trap type, 8 contracts.**

---

## After Annotation: Running the Evaluation

Once your ground truth is ready, everything is automated:

```bash
# Step 1: Run batch evaluation (takes ~30-60 min due to API calls)
python -m evaluation.batch_evaluation

# Step 2: Run statistical analysis on the results
python -m evaluation.statistical_analysis results/eval_results_XXXXXXXX_XXXXXX.json

# Step 3 (optional): Generate LaTeX table for your paper
python -m evaluation.statistical_analysis results/eval_results_XXXXXXXX_XXXXXX.json --latex
```

---

## Quality Checklist

Before running the evaluation, verify:

- [ ] 48 entries total (8 contracts × 6 queries)
- [ ] 16 entries per trap type (trap_a, trap_b, trap_c)
- [ ] Each ground truth answer includes specific section references
- [ ] Each entry has 3-8 information units
- [ ] Ground truth answers are 50-300 words each
- [ ] No REPLACE placeholders remaining
- [ ] `python -m data.ground_truth_loader` reports no issues
