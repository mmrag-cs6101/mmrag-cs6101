# Investigation Findings Summary

**Date:** October 4, 2025
**Issue:** 1.5% accuracy on Sprint 10 validation (target: 53-59%)

---

## 🔍 What I Discovered

### 1. **Terminology Confusion** (RESOLVED)

The Sprint 10 results use confusing terminology:
- "successful_queries" = queries with CORRECT answers
- "failed_queries" = queries with INCORRECT answers
- "error_rate" = WRONG ANSWER rate (NOT system crash rate)

**Reality**: The system is NOT crashing or failing. It completes all 4,059 queries successfully, but only 61 (1.5%) produce correct answers.

### 2. **Multiple-Choice Assumption** (INCORRECT - NOW REVERTED)

I initially assumed MRAG-Bench was a multiple-choice task (A/B/C/D). This was **WRONG**.

**Actual Task Format:**
- Open-ended question answering
- Answers are free-form text (e.g., "hartebeest", "Versailles", "silky_terrier")
- No multiple-choice options in the dataset
- Evaluation uses Jaccard similarity (>= 0.6) and medical keyword matching (>= 0.7)

**What I Did:**
- ✅ Reverted all multiple-choice changes
- ✅ Restored original open-ended prompt
- ✅ Restored full generation parameters (not constrained to 5 tokens)
- ✅ Removed A/B/C/D answer extraction

### 3. **Conflicting Test Results** (KEY MYSTERY)

There are TWO different test results with VERY different outcomes:

#### Result Set A: `output/real_mrag_test/test_results.json`
```
Accuracy: 100% (45/45 correct)
Sample outputs:
  - Q: "Can you identify this animal?"
  - GT: "silky_terrier"
  - Generated: "silky_terrier" ✅

  - Q: "Which creature is this?"
  - GT: "impala"
  - Generated: "impala" ✅
```

#### Result Set B: `output/sprint10/sprint10_final_validation_results.json`
```
Accuracy: 1.5% (20/1353 correct)
Scenario breakdown:
  - Angle: 0.2% (1/544)
  - Partial: 0.8% (3/348)
  - Scope: 4.7% (17/353)
  - Occlusion: 0.0% (0/108)
```

**Why the Huge Difference?**

Possible explanations:
1. **Different test sets**: real_mrag_test used easier/curated samples
2. **Code changes between runs**: Something broke between the two evaluations
3. **Configuration differences**: Different prompts, parameters, or retrieval settings
4. **Retrieval quality**: Sprint 10 might have poor retrieval quality

---

## 🎯 Root Cause Hypotheses

Based on the evidence, the 1.5% accuracy is likely caused by ONE of these:

### Hypothesis A: Poor Retrieval Quality
**Symptoms:**
- Retriever returns irrelevant images
- LLaVA generates answers based on wrong context
- Answers don't match ground truth

**How to Test:**
```bash
# Check retrieval quality for a known sample
python -c "
from src.pipeline import MRAGPipeline
from src.config import MRAGConfig

config = MRAGConfig.load('config/mrag_bench.yaml')
pipeline = MRAGPipeline(config)
pipeline.load_retriever()

# Test retrieval
results = pipeline.retriever.retrieve_similar('Can you identify this animal?', k=5)
for idx, (path, score) in enumerate(results):
    print(f'{idx+1}. {path} (score: {score:.3f})')
"
```

### Hypothesis B: Prompt Engineering Issues
**Symptoms:**
- Model generates verbose/wrong format responses
- Responses don't contain the key answer terms
- Jaccard similarity < 0.6 even when conceptually correct

**How to Test:**
Run quick evaluation and check actual outputs

### Hypothesis C: Evaluation Criteria Too Strict
**Symptoms:**
- Generated answers are semantically correct
- But don't pass Jaccard >= 0.6 or keyword >= 0.7 thresholds
- Examples: "dog" vs "canine", "heart attack" vs "myocardial infarction"

**How to Test:**
Lower thresholds temporarily and re-evaluate

---

## 🚀 Recommended Action Plan

### Step 1: Run Diagnostic Test (5 minutes)
```bash
source mrag-bench-env/bin/activate
python run_sprint10_final_validation.py --quick-test --max-samples 10
```

Look at the output to see:
- What answers are being generated
- How they compare to ground truth
- Whether they're close matches or completely wrong

### Step 2: Based on Findings, Apply Fix

#### If answers are completely irrelevant:
**Problem**: Retrieval returning wrong images
**Fix**:
- Check corpus embedding quality
- Increase retrieval top-k from 4 to 10
- Verify FAISS index is correct

#### If answers are close but not matching:
**Problem**: Prompt or generation parameters
**Fix**:
- Improve prompt to request concise answers
- Adjust temperature (lower for more deterministic)
- Add examples in prompt (few-shot learning)

#### If answers are correct but failing criteria:
**Problem**: Evaluation matching too strict
**Fix**:
- Lower Jaccard threshold from 0.6 to 0.4
- Improve normalization (synonyms, medical terms)
- Use semantic similarity instead of word overlap

### Step 3: Iterate and Validate
```bash
# After each fix, run quick test
python run_sprint10_final_validation.py --quick-test --max-samples 50

# When accuracy improves, run full validation
python run_sprint10_final_validation.py --num-runs 3
```

---

## 📝 Current Status

### ✅ What's Been Fixed
1. Reverted incorrect multiple-choice changes
2. Restored original open-ended QA format
3. Understood evaluation methodology
4. Clarified "error_rate" terminology confusion

### ❓ What's Still Unknown
1. Why Sprint 10 validation had 1.5% vs earlier 100% accuracy
2. What specific aspect is causing low accuracy (retrieval/generation/evaluation)
3. What actual answers are being generated in the failing cases

### 🔄 Next Immediate Step
**Run a quick diagnostic test** with 10 samples to see actual generated outputs and identify the real bottleneck.

---

## 💡 Key Insights

1. **The system works** - it's not crashing, just producing wrong answers
2. **MRAG-Bench is open-ended QA**, not multiple choice
3. **There's a discrepancy** between different test runs that needs investigation
4. **Need to see actual outputs** before applying fixes

---

**Recommendation**: Before implementing any "fixes", run the quick diagnostic test to see what's actually being generated. Fixing the wrong problem will waste time and potentially make things worse.
