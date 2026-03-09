# CM2US Use Case — Critical Minerals Closed-Loop Research

## Strategic Context

**CM2US** (Critical Minerals and Materials to Unlock Supply) is a DOE initiative
led by 12 national laboratories (Ames Lab as lead) under the Genesis Mission.
The strategic urgency: the US is nearly 100% import-dependent on many critical
minerals, and China controls ~80% of global REE processing capacity.
CM2US is the national-scale response.

**Genesis Mission framing**: Use AI, advanced computing, and next-generation
research platforms to revolutionize US science and innovation. CM2US is the
critical minerals instantiation of that mission — running "hundreds of experiments
in days instead of years."

**The specific promise of closed-loop research here**: automated systems running
24/7, AI proposing and testing its own ideas, removing the delays of manual labor.
This is exactly what we're building.

---

## The Three Research Pillars

The CM2US workflow runs on three sequential but interconnected pillars:

```
PROSPECTING          →     PROCESSING           →     PRODUCTION
──────────────────         ──────────────────         ──────────────────
Where are the              How do we extract          Can we do it at
minerals hiding?           them efficiently?          pilot scale profitably?

XRF, ICP-MS data           Leaching experiments       Hydromet pilot plants
Geochemical models         Recovery optimization      Economic modeling
Site classification        Separation chemistry       Process scale-up
Basin comparisons          pH / temp / acid tuning    Purity thresholds
```

**The closed-loop system lives primarily in Processing**, where the search space
is well-defined, experiments are expensive, and nonlinear effects dominate.
Prospecting feeds the inputs; Production consumes the outputs.

---

## Feedstocks Being Studied

Coal-based unconventional sources are the primary domestic REE opportunity:

| Feedstock | Annual Volume | Median REE Content | Notes |
|---|---|---|---|
| Coal fly ash | ~38M tons/yr | 481 mg/kg | Most studied, best characterized |
| Coal bottom ash | ~9M tons/yr | ~300-400 mg/kg | Less enriched than fly ash |
| Coal refuse / reject | Varies | Variable | Abandoned mine sites |
| Acid mine drainage (AMD) | Continuous flow | Low ppm, high volume | Treatment solids more enriched |
| AMD treatment solids | Varies | ~1000-5000 mg/kg | Concentrated by precipitation |
| Clay/shale overburden | Vast | Variable | Undercharacterized |

---

## Target Minerals (Priority Order)

### Heavy REEs (highest strategic value, lowest global supply security)
- **Dysprosium (Dy)** — permanent magnets for EV motors and wind turbines
- **Terbium (Tb)** — green phosphors, EV motor magnets
- **Erbium (Er)** — fiber optic amplifiers
- **Yttrium (Y)** — phosphors, ceramics, superconductors

### Light REEs (high volume demand)
- **Neodymium (Nd)** — dominant in NdFeB permanent magnets (EVs, wind)
- **Europium (Eu)** — red phosphors, lighting
- **Cerium (Ce)** — catalysts, polishing
- **Lanthanum (La)** — batteries, optics

### Other Critical Minerals (expanding focus)
- **Cobalt** — EV batteries; CM2US using synthetic biology (peptide binding) for Co
- **Lithium** — batteries
- **Scandium** — aerospace alloys
- **Germanium** — semiconductors, fiber optics

---

## The Researcher's Actual Workflow

What a CM2US researcher does day-to-day — and where the loop lives:

```
STAGE 1: SAMPLE CHARACTERIZATION
────────────────────────────────────────────────────────────────────
Collect coal ash / AMD / ore sample from mine site or power plant
Run handheld XRF (field-deployable, fast) → major element profile
  Output: Al, Si, Fe, Ca, Ti, K, Mg, Na concentrations
Run ICP-MS (lab, slower, precise) → full trace REE concentrations
  Output: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Y, Sc
Compare: can XRF proxy for ICP-MS? (if yes → huge field efficiency gain)

STAGE 2: FEEDSTOCK SCREENING
────────────────────────────────────────────────────────────────────
Hypothesis: "Which source material is most enriched / worth processing?"
Compare multiple samples across basins, sites, ash ponds
ML model: predict ICP-MS REE concentrations from XRF major elements
  Key ML task: regression, features=major oxides, target=ΣREE or individual REE
Rank feedstocks by HREE content, accessibility, volume

STAGE 3: LEACHING OPTIMIZATION  ← PRIMARY CLOSED-LOOP TARGET
────────────────────────────────────────────────────────────────────
Take enriched feedstock → run leaching experiments
Vary conditions → measure what comes out
Multiple iterations required: nonlinear effects, interaction terms
AutoML runs predict recovery % from condition parameters
Acquisition function proposes next conditions to test
This is where our closed-loop system plugs in directly

STAGE 4: SEPARATION & PURIFICATION
────────────────────────────────────────────────────────────────────
Solvent extraction: di(2-ethylhexyl)phosphoric acid in kerosene/mineral oil
Oxalic acid precipitation to concentrate REEs from leachate
Calcination → rare earth oxides (final product: >90% purity target)
Stripping: 5M nitric acid solution

STAGE 5: SCALE-UP DECISION
────────────────────────────────────────────────────────────────────
Is yield high enough at pilot scale (target: 1-3 tonnes/day mixed REO)?
Economic modeling: profitable at $X/kg REE given energy + reagent costs?
Pilot facilities currently producing ~100 gm/day → scaling to tonnes/day
```

---

## The Leaching Variable Space

This is the core experimental space the closed-loop system searches.

### Independent Variables (what the researcher controls)

| Variable | Typical Range | Key Insight |
|---|---|---|
| **Acid concentration** | 0.5M – 6M HCl (or H₂SO₄) | Single most important factor |
| **Temperature** | 25°C – 95°C | Nonlinear: plateau ~65-80°C |
| **Leaching time** | 30 min – 6 hours | Diminishing returns past optimum |
| **pH (initial/final)** | 0.5 – 5.0 | Critical threshold: >pH 4 causes REE loss |
| **Liquid-to-solid ratio** | 5:1 – 50:1 | Dilution vs. recovery tradeoff |
| **Particle size** | <75μm – >250μm | Surface area → higher yield |
| **Acid type** | HCl, H₂SO₄, HNO₃, citric | HCl most effective, citric = green alternative |
| **Oxidizer presence** | H₂O₂ added or not | Affects certain element valence states |

### Dependent Variables (what the researcher measures)

| Variable | Unit | Optimization goal |
|---|---|---|
| **REE recovery %** | % of input extracted | Maximize — primary target |
| **Individual element yield** | % (Nd, Dy, Eu separately) | Element-specific maximization |
| **Total REE in leachate** | mg/L or ppm | Concentration measure |
| **Nd recovery %** | % | Phase 1 specific target |
| **Dy recovery %** | % | High value HREE target |
| **Al co-extraction** | ppm | Minimize — purity constraint |
| **Fe co-extraction** | ppm | Minimize — purity constraint |
| **Purity** | % REE / total dissolved solids | Constraint: >90% for product |
| **Energy consumption** | kWh/kg REE produced | Economic constraint |

### Known Domain Priors (encode in acquisition function)

These come from published literature and should seed the system's knowledge:

```
CONFIRMED EFFECTS:
• HCl concentration is the dominant predictor of total REE recovery
• Optimal HCl: 2.5-4M for coal fly ash (beyond 4M: Al co-extraction spike)
• Optimal temperature: 55-75°C (diminishing returns above 80°C)
• Optimal leaching time: 2-3 hours for most fly ash feedstocks
• pH > 4 is a hard constraint — REE precipitation losses become severe
• HREE and LREE respond differently to same conditions (need separate optimization)

KNOWN INTERACTIONS:
• Temperature × time: synergistic effect in 55-75°C / 2-3hr region
• Acid concentration × particle size: fine particles amplify acid effect
• pH drift during leaching: initial pH != final pH in acid-rock reactions

FEEDSTOCK-SPECIFIC EFFECTS (memory-worthy):
• Appalachian ash: higher HREE relative to LREE vs Gulf Coast
• AMD precipitates: different optimal conditions than fly ash
• Higher initial Al/Fe in sample → more impurity co-extraction at given acid conc.

ECONOMIC CONSTRAINTS:
• Energy cost dominates at temperature > 80°C
• Acid neutralization cost scales with acid concentration
• Target: >80% REE recovery at <$XX/kg energy+reagent cost
```

---

## Reported Benchmark Performance (Literature)

What good results look like — success criteria for the system:

| Condition | Nd | Er | Eu | Tb | Dy |
|---|---|---|---|---|---|
| 3M HCl, 65°C, 270 min | 70.8% | 76.3% | 88.0% | 90.0% | 73.4% |
| Oxalic acid (green alt.) | up to 91.5% Nd | — | — | — | 81.8% Dy |
| AMD treatment solids | varies | — | — | — | varies |

Target for CM2US pilot scale: **>90% purity mixed REO**, **>80% individual REE recovery**.

---

## How the Existing Stack Maps to CM2US

Every piece already built has a direct role:

| CM2US Research Need | Your Stack Component | Status |
|---|---|---|
| Experimental data store | `dev_europa.gold_roses` (32 tables) | Live |
| Leaching result queries | DatabricksCatalogAgent (15 tools) | Live |
| Literature on REE recovery | GoogleScholarAgent | Live |
| NETL public REE datasets | EDXDataDiscoveryAgent | Live |
| Predict recovery from conditions | AutoMLAgent (inner loop) | Live |
| Domain interpretation | Fine-tuned Mistral-7B | Trained (endpoint pending) |
| Track experiment runs | MLflow | Live |
| Memory of what's been tried | Scientific Memory (planning) | To build |
| NL hypothesis → config | Hypothesis Parser (planning) | To build |
| What to try next | Acquisition Function (planning) | To build |

### The EDX Connection is Especially Powerful

NETL EDX (already integrated) hosts:
- REE characterization data from coal ash across US basins
- AMD chemistry datasets
- XRF and ICP-MS measurement compilations
- Geochemical profiles of coal refuse sites
- Critical minerals concentration maps

The closed-loop system can pull these reference datasets before each iteration
to benchmark your gold_roses results against the national dataset. If your
leaching conditions yield 78% Nd recovery and the EDX dataset shows similar
feedstocks averaging 65% — the system knows you're ahead of the field.

---

## Five Hypothesis Types for CM2US Researchers

These are the natural language inputs the system should handle:

### 1. Recovery Optimization (Phase 1 target)
```
"I want to optimize leaching conditions for Neodymium recovery
 from our coal fly ash samples. I think acid concentration is
 the primary driver. Maximize recovery while keeping Al < 500 ppm."
```

### 2. Source Screening
```
"Which of our coal ash samples has the highest HREE concentration?
 Can I predict ICP-MS values from XRF readings to avoid expensive
 lab analysis for initial screening?"
```

### 3. Feedstock Generalization
```
"Does the optimal leaching protocol we found for Appalachian ash
 transfer to Gulf Coast samples, or do we need feedstock-specific
 protocols? Test on both sample sets."
```

### 4. Constrained Multi-Objective
```
"Find the acid concentration and temperature that maximize Dy recovery
 subject to: Al co-extraction < 500 ppm, energy cost < $X/kg,
 leaching time < 2 hours."
```

### 5. Proxy Validation
```
"Test whether handheld XRF major element readings (Al, Fe, Si, Ca, Ti)
 can predict ICP-MS REE concentrations well enough to replace ICP-MS
 for field screening. How accurate does the model get?"
```

---

## Phase 1 Specific Use Case: Nd Recovery Optimization

**The concrete first implementation:**

```
Hypothesis:  "Acid concentration is the primary driver of Neodymium
              recovery from coal fly ash. Optimize conditions for
              maximum Nd recovery with Al co-extraction < 500 ppm."

Data source: dev_europa.gold_roses.[leaching_results_table]

Target:      nd_recovery_pct  (or equivalent column name)

Key IVs:     hcl_concentration_M
             temperature_C
             leaching_time_min
             ph_final
             liquid_solid_ratio

Constraints: al_coextraction_ppm < 500

Literature:  GoogleScholarAgent → "HCl coal fly ash Nd recovery optimization"
             Known prior: optimal HCl ~2.5-4M, temp ~65-75°C

EDX context: Pull NETL REE coal ash characterization datasets for
             benchmark comparison of recovery performance

AutoML task: regression
             features: [hcl_conc, temp, time, ph, ls_ratio, feedstock_type]
             target: nd_recovery_pct

Success:     nd_recovery_pct > 80% AND al_coextraction_ppm < 500

Expected iterations to convergence: 5-8 runs
```

---

## Acquisition Function Priors for CM2US

The acquisition function should encode these domain constraints as hard and soft rules:

```python
HARD CONSTRAINTS (never propose these):
  ph_final > 4.5                   # REE precipitation — guaranteed loss
  temperature_C > 95               # Equipment limits + energy cost cliff
  hcl_concentration_M > 6         # Diminishing returns + safety + cost

SOFT CONSTRAINTS (penalize in scoring):
  temperature_C > 80               # Energy cost increases sharply
  leaching_time_min > 360         # 6hrs+ impractical at scale
  hcl_concentration_M > 4.5       # Al co-extraction spike risk

PRIORITY REGIONS (favor exploring):
  hcl_conc: 2.5-4.0 AND temp: 55-75 AND time: 90-180   # Literature sweet spot
  interaction: temperature × time (underexplored per literature)

ANOMALY FLAGS (pause for user review):
  nd_recovery_pct > 92%            # Suspiciously high — check for data error
  nd_recovery_pct < 20%           # Much worse than baseline — investigate
  al_coextraction_ppm > 2000      # Purity problem — surface immediately
```

---

## NETL EDX Datasets to Query

Specific EDX search terms for the context layer (pre-experiment grounding):

```python
EDX_SEARCH_QUERIES = [
    "rare earth element coal fly ash characterization",
    "REE concentration coal ash ICP-MS XRF",
    "acid mine drainage rare earth recovery",
    "critical minerals leaching coal byproducts",
    "neodymium dysprosium recovery coal refuse",
    "HREE concentration appalachian coal basin",
    "REE critical mineral geochemistry sedimentary"
]
```

These feed the Search Context Layer before each iteration, pulling published
characterization data to benchmark against and contextualize findings.

---

## Scientific Memory Seeds

Pre-populate the memory store with known literature findings so the system
starts informed rather than naive:

```python
SEED_FINDINGS = [
    {
        "finding_text": "HCl concentration is the dominant predictor of total REE recovery from coal fly ash. Optimal range 2.5-4M for most feedstocks.",
        "confidence": 0.90,
        "finding_type": "predictor",
        "variables": ["hcl_concentration_M", "ree_recovery_pct"],
        "tags": ["literature", "fly_ash", "leaching"]
    },
    {
        "finding_text": "pH above 4 causes iron and aluminum hydroxide precipitation, co-precipitating REEs and causing significant recovery losses.",
        "confidence": 0.95,
        "finding_type": "threshold",
        "variables": ["ph_final", "ree_recovery_pct"],
        "tags": ["literature", "hard_constraint", "precipitation"]
    },
    {
        "finding_text": "Temperature effect on REE recovery is nonlinear: yield increases to ~65-80°C then plateaus or degrades. Energy cost increases sharply above 80°C.",
        "confidence": 0.85,
        "finding_type": "threshold",
        "variables": ["temperature_C", "ree_recovery_pct"],
        "tags": ["literature", "nonlinear", "energy_cost"]
    },
    {
        "finding_text": "HREE (Dy, Tb, Er) and LREE (Nd, Ce, La) respond differently to the same leaching conditions. Separate optimization may be needed.",
        "confidence": 0.80,
        "finding_type": "interaction",
        "variables": ["hree_recovery_pct", "lree_recovery_pct"],
        "tags": ["literature", "element_selectivity"]
    },
    {
        "finding_text": "At 3M HCl, 65°C, 270 minutes: Nd 70.8%, Er 76.3%, Eu 88.0%, Tb 90.0%, Dy 73.4% recovery achieved from Philippine coal fly ash.",
        "confidence": 0.95,
        "finding_type": "predictor",
        "variables": ["hcl_concentration_M", "temperature_C", "leaching_time_min", "nd_recovery_pct"],
        "tags": ["literature", "benchmark", "fly_ash"]
    }
]
```

---

## Success Metrics Aligned with CM2US Program Goals

| Metric | CM2US Target | Phase 1 System Goal |
|---|---|---|
| REE recovery % | >80% individual elements | System finds conditions achieving >80% |
| Product purity | >90% mixed REO | Al co-extraction constraint respected |
| Experiment efficiency | 100x vs manual | 5-8 loop iterations vs 40+ manual runs |
| Time to optimum | Days not years | Session completes in hours not weeks |
| Knowledge transfer | Findings shared across labs | Scientific memory persists across sessions |
| Feedstock coverage | Coal, AMD, refuse, shale | Start with fly ash, expand by feedstock type |

---

## Open Questions to Resolve Before Building

1. **Column names in gold_roses**: What are the actual column names for
   the leaching variables? Run `DatabricksCatalogAgent.profile_dataset()`
   on the relevant tables to confirm exact schema before Phase 1 build.

2. **Which tables are leaching results?**: Identify the specific gold_roses
   tables that contain leaching experiment data vs. characterization data
   vs. metadata. Table names + row counts will drive the Hypothesis Parser's
   table inference.

3. **Target column availability**: Is `nd_recovery_pct` (or equivalent)
   actually in the data? Or is recovery calculated from input/output concentration?

4. **Feedstock metadata**: Is there a column identifying sample source
   (Appalachian, Gulf Coast, AMD, etc.)? This enables feedstock-specific models.

5. **Experiment ID linking**: Are individual leaching runs identified by a
   unique experiment ID that links input conditions to output measurements?
   Critical for AutoML feature construction.
