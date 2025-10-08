# MRAG-Bench Accuracy Analysis - Corrected

**Current Status:** 1.5% success rate (NOT accuracy issue - it's a FAILURE issue)
**Target:** 53-59% accuracy
**Date:** October 4, 2025

---

## ❌ INCORRECT INITIAL DIAGNOSIS

The original improvement plan assumed MRAG-Bench was a multiple-choice task. **This was wrong.**

### What I Initially Thought:
- Task: Multiple-choice (A/B/C/D)
- Problem: Model generates explanations instead of letters
- Solution: Constrain output to force A/B/C/D selection

### Actual Reality:
- **Task**: Open-ended question answering
- **Dataset**: No multiple-choice options in the data
- **Answers**: Free-form text (e.g., "hartebeest", "Versailles", "macaque")
- **Problem**: 98.5% of queries are **FAILING**, not producing wrong answers

---

## 🔍 Root Cause Analysis (Corrected)

### Critical Issue: Massive Query Failure Rate

**Evidence from sprint10_final_validation_results.json:**
```json
{
  "total_queries": 4059,
  "successful_queries": 61,
  "failed_queries": 3998,
  "success_rate": 0.015028332101502834,  // 1.5%
  "error_rate": 0.9849716678984972       // 98.5%
}
```

**The 1.5% "accuracy" is actually:**
- Only 61 successful generations out of 4059 attempts
- NOT that the model gets 98.5% wrong
- But that the model FAILS to generate 98.5% of the time

### Success Breakdown by Scenario:
- **Angle**: 0.2% success (1 out of 544 samples)
- **Partial**: 0.8% success (3 out of 348 samples)
- **Scope**: 4.7% success (17 out of 353 samples) ← Best performer
- **Occlusion**: 0.0% success (0 out of 108 samples) ← Complete failure

---

## 🎯 Real Problem: Why Are Queries Failing?

### Hypothesis 1: Generation Timeout/OOM
- Generation time: ~0.9-1.1s per query (within limits)
- Memory usage: 4.1GB / 16GB (fine)
- **Unlikely** - performance metrics look good

### Hypothesis 2: Prompt Format Issues
- Current prompt includes `<image>` tokens
- May have mismatch between prompt format and model expectations
- Need to check error logs for actual failure messages

### Hypothesis 3: Retrieval Returning No Images
- If retrieval fails to find images, generation might fail
- Retrieval time: 0.01s (very fast, might be returning empty results)
- **LIKELY** - retrieval might be failing silently

### Hypothesis 4: Post-Processing Rejecting Outputs
- Model generates valid text
- Post-processing or evaluation logic marks it as "failed"
- Need to check what constitutes a "failed" query

---

## 🔬 Diagnostic Steps Required

### Step 1: Check Actual Error Messages
```bash
# Need to find logs or error tracking
grep -r "error\|Error\|ERROR" output/sprint10/
tail -100 logs/evaluation.log  # if it exists
```

### Step 2: Test Single Query End-to-End
```python
# Run one complete query with detailed logging
from src.pipeline import MRAGPipeline
from src.config import MRAGConfig

config = MRAGConfig.load('config/mrag_bench.yaml')
pipeline = MRAGPipeline(config)

# Test with known sample
result = pipeline.process_query(
    question="Can you identify this animal?",
    question_id="mrag_000007",  # Known "angle" sample
    ground_truth="hartebeest"
)

print(f"Success: {result is not None}")
if result:
    print(f"Answer: {result.answer}")
    print(f"Metadata: {result.metadata}")
```

### Step 3: Check Retrieval Output
```python
# Verify retrieval is working
retriever = pipeline.retriever
query_embedding = retriever.encode_text("Can you identify this animal?")
retrieved = retriever.search(query_embedding, top_k=5)

print(f"Retrieved {len(retrieved)} images")
for idx, (image_path, score) in enumerate(retrieved):
    print(f"  {idx+1}. {image_path} (score: {score:.3f})")
```

### Step 4: Check Generation Directly
```python
# Bypass pipeline, test generation directly
from src.generation import LLaVAGenerationPipeline, MultimodalContext
from PIL import Image

gen_pipeline = LLaVAGenerationPipeline(config.model)
gen_pipeline.load_model()

# Load test image
images = [Image.open("data/mrag_bench/images/image_000007.jpg")]

context = MultimodalContext(
    question="Can you identify this animal?",
    images=images
)

result = gen_pipeline.generate_answer(context)
print(f"Generated: {result.answer}")
print(f"Confidence: {result.confidence_score}")
print(f"Metadata: {result.metadata}")
```

---

## 💡 Likely Solutions (Based on Common Issues)

### If Retrieval is Failing:
1. **Check corpus embeddings**: Ensure all 14,475 images were embedded
2. **Verify FAISS index**: Index might be corrupted or empty
3. **Fix retrieval logic**: May be returning empty lists

### If Generation is Failing:
1. **Check input validation**: Model might reject certain prompts
2. **Fix image preprocessing**: Images might not be in correct format
3. **Adjust generation parameters**: Temperature/top_p might be too restrictive

### If Evaluation is Rejecting Outputs:
1. **Check answer matching logic**: Too strict string matching
2. **Fix evaluation criteria**: Might be marking valid answers as failures
3. **Adjust confidence thresholds**: Might be filtering out low-confidence answers

---

## 📊 Expected Impact After Fixes

If we fix the failure rate:

| Scenario | Success Rate Fix | Expected Accuracy After Fix |
|----------|------------------|---------------------------|
| Angle | 0.2% → 100% | 30-50% (with proper retrieval) |
| Partial | 0.8% → 100% | 35-55% |
| Scope | 4.7% → 100% | 50-65% (already best) |
| Occlusion | 0.0% → 100% | 25-45% |

**Rationale**: Once queries stop failing, we should get baseline LLaVA + CLIP performance, which typically achieves 30-60% on visual QA tasks.

---

## 🚀 Corrected Action Plan

### Priority 1: Diagnose Failure Cause
1. Run diagnostic Step 1-4 above
2. Check logs for actual error messages
3. Test individual components in isolation

### Priority 2: Fix Root Cause
- If retrieval: Fix embedding/indexing
- If generation: Fix prompt/preprocessing
- If evaluation: Fix matching logic

### Priority 3: Validate Fix
```bash
# Quick test after fix
source mrag-bench-env/bin/activate
python run_sprint10_final_validation.py --quick-test --max-samples 10
```

### Priority 4: Optimize for Accuracy
- ONLY after success rate >95% should we work on accuracy
- Then apply prompt engineering, parameter tuning, etc.

---

## ❌ What NOT to Do

1. ❌ Don't add multiple-choice logic (dataset has no choices)
2. ❌ Don't constrain generation to 5 tokens (need free-form answers)
3. ❌ Don't add A/B/C/D extraction (answers are "hartebeest", not "A")
4. ❌ Don't optimize accuracy before fixing failure rate

---

## ✅ Next Immediate Steps

1. **Revert incorrect multiple-choice changes** ← DONE
2. **Run diagnostic tests** to find actual error
3. **Fix the root cause** of 98.5% failure rate
4. **Validate** that success rate reaches >95%
5. **Then** optimize for accuracy improvements

---

**Status**: Ready to diagnose the actual problem
