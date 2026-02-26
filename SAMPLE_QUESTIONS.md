# Sample Questions for Each Training Stage

This document provides example questions to test your model at each stage of training. It helps you understand what the model can and cannot do at each phase, and verify that training is working correctly.

---

## Base Mistral-7B (Before Any Training)

The base model knows nothing about your data. These questions establish a baseline — the model will either hallucinate or give generic answers.

### Questions to Ask
```
1. "What tables exist in the gold_roses schema?"
2. "What is the typical analytical response for oxide samples?"
3. "Describe the relationship between temperature and pressure in our experimental data."
4. "What columns does the sample_properties table have?"
5. "What is the average temperature across all experiments?"
```

### What You Should Expect
- Generic or completely made-up answers
- No knowledge of your tables, columns, or values
- May say something like "I don't have access to your database"
- Or worse, confidently hallucinate fake data

### Why This Matters
This is your control group. Compare these answers to the same questions after CPT training to see the improvement.

---

## After CPT Training (Phase 1)

The model has absorbed your domain data as natural language passages. It understands the vocabulary, structure, and patterns but wasn't trained to answer questions directly.

### Good Questions (model should handle reasonably)

**Domain vocabulary and structure:**
```
1. "Describe the data in the gold_roses schema."
2. "What types of measurements are recorded in our experimental data?"
3. "What fields are commonly found across the tables in gold_roses?"
4. "Continue this passage: Record from Gold Roses — Identification —"
5. "What kinds of materials appear in our dataset?"
```

**Pattern completion (the model's strongest skill after CPT):**
```
6. "A typical oxide sample has measurements including"
7. "The gold_roses dataset contains records with properties such as"
8. "Data from the experimental analysis shows that temperature values"
9. "The relationship between sample properties and analytical results"
10. "Shared columns across tables in the schema include"
```

### Tricky Questions (model may struggle)

**Direct lookups — it absorbed patterns, not a database:**
```
11. "What was the exact analytical response for sample S-001?"
12. "How many rows are in the sample_properties table?"
13. "List all sample IDs where temperature exceeds 300."
```

**Structured analysis — it wasn't trained to reason about data:**
```
14. "Compare oxide vs sulfide materials across all metrics."
15. "Which model should I deploy for neutron flux prediction?"
16. "Give me a recommendation with justification and risks."
```

### What You Should Expect
- Questions 1-5: Decent answers that reference real column names and table structures from your data
- Questions 6-10: Natural completions that sound like your domain — this is where CPT shines
- Questions 11-13: May get lucky on common patterns, but won't be a reliable lookup
- Questions 14-16: Rambling, unstructured, or generic answers — this is what SFT fixes

### How to Tell If CPT Worked
Compare the CPT model's answer to the base model's answer for the same question. You should see:
- Real table names from your schema appearing in responses
- Real column names (temperature, pressure, analytical_response, etc.)
- Value ranges that match your actual data
- Domain-specific phrasing that wasn't there before

### Red Flags (something went wrong)
- Model outputs gibberish or repeating tokens → training was too aggressive or data was corrupted
- Answers are identical to the base model → model didn't learn (check data loading)
- Model only outputs raw column values with no context → data conversion to natural language didn't work properly

---

## After SFT Training (Phase 3)

The model now knows your domain AND can answer questions in a structured, helpful way.

### Good Questions (model should handle well)

**Row-level (direct data questions):**
```
1. "What was the analytical response for sample S-001?"
2. "Tell me about the measurements for sample S-042."
3. "What material type is sample S-017?"
4. "What properties were measured for the most recent oxide sample?"
```

**Aggregation (computed answers):**
```
5. "What is the average temperature for oxide samples?"
6. "What is the range of pressure values across all experiments?"
7. "How many sulfide samples are in the dataset?"
8. "Which material type has the most samples?"
```

**Comparison (cross-group analysis):**
```
9. "How does analytical response differ between oxide and sulfide materials?"
10. "Compare temperature distributions across material types."
11. "Which material type shows the highest average pressure?"
12. "Are there any outliers in the analytical response data?"
```

**Schema-level (structural understanding):**
```
13. "What tables are available in the gold_roses schema and how are they related?"
14. "Which columns are shared across multiple tables?"
15. "Describe the purpose of each table in the schema."
16. "How many total records exist across all tables?"
```

**Reasoning and recommendation:**
```
17. "Which model should I deploy for predicting analytical response in oxide materials?"
18. "What are the risks of using the linear model for real-time predictions?"
19. "If inference latency must be under 20ms, which model should I choose?"
20. "Compare the trade-offs between the neural network and XGBoost models."
```

### What You Should Expect
- Structured, direct answers that reference actual data
- Correct values (not hallucinated) for questions about your dataset
- Clear recommendations with justification when asked
- Responses formatted in a readable way (not rambling text)

### How to Tell If SFT Worked
Compare the same question across all three stages:

**Question: "What is the average temperature for oxide samples?"**

| Stage | Expected Response |
|---|---|
| Base Mistral | "I don't have access to your data" or makes up a number |
| After CPT | Mentions temperature and oxide but gives a vague or rambling answer |
| After SFT | "Oxide samples have an average temperature of 305.2 across 47 measurements, ranging from 198.4 to 412.1." |

**Question: "Describe the gold_roses schema."**

| Stage | Expected Response |
|---|---|
| Base Mistral | "I'm not familiar with that schema" or generic SQL advice |
| After CPT | Mentions some real table names and columns but in an unstructured way |
| After SFT | Clean summary listing tables, their purposes, row counts, and relationships |

---

## Quick Test Script

After each training phase, run these 5 questions as a consistent benchmark:

```
1. "What tables exist in the gold_roses schema?"
2. "What is the average temperature for oxide samples?"
3. "Compare oxide and sulfide materials."
4. "Which model should I deploy for analytical response prediction?"
5. "Describe the relationship between temperature and analytical response."
```

Save the outputs from each phase side by side. The progression from base → CPT → SFT should show a clear improvement in both domain accuracy and response quality.

---

## Tips for Testing

- **Ask the same question multiple times** — if you used `do_sample=True` (which we do), responses will vary. Ask 2-3 times to get a sense of consistency.
- **Try breaking it** — ask about data that doesn't exist in your tables. A well-trained model should say it doesn't know rather than hallucinate.
- **Test edge cases** — ask about column names that are similar but not exact. Does it correct you or go along with the wrong name?
- **Compare response length** — CPT models tend to ramble. SFT models should give concise, structured answers. If your SFT model is still rambling, you may need more training steps or better Q&A examples.
