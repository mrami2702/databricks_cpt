# SFT Data Generation Fix — Subgroup StdDev None Error

**Problem:** `unsupported format string passed to NoneType.__format__` in the subgroup Q&A step (Step 14).

**Root cause:** When a categorical group has only 1 sample, Spark returns `None` for stddev. The f-string tried to format `None` with `:.6g`.

**Fix:** In `generate_sft_data.py`, find this line in the subgroup step (~line 1235):

```python
# OLD — crashes when r['std'] is None
f"'{str(r[cat_col])}': mean {r['avg']:.6g}, std {r['std']:.6g if r['std'] else 0:.6g}, n={r['cnt']}"
```

Change to:

```python
# NEW — wraps ternary in parentheses so format applies to the result
f"'{str(r[cat_col])}': mean {r['avg']:.6g}, std {(r['std'] if r['std'] is not None else 0):.6g}, n={r['cnt']}"
```

**Why:** Python was parsing `{r['std']:.6g if ...}` as trying to format `r['std']` directly with `:.6g`, then hitting the `if` as a syntax surprise. Wrapping in parentheses `{(expr):.6g}` forces Python to evaluate the ternary first, then format the result.
