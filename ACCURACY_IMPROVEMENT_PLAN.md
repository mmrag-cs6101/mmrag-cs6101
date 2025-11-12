# MRAG-Bench Accuracy Improvement Plan

**Current Status:** 1.5% accuracy (Target: 53-59%)
**Gap:** 51.5 percentage points below minimum target
**Date:** October 4, 2025

---

## 🔍 Root Cause Analysis

### Critical Issue: Multiple Choice Format Not Implemented

**Problem:** The system is treating MRAG-Bench as an open-ended QA task, but it's actually a **multiple-choice question answering task**.

**Evidence:**
- Current accuracy: 1.5% (essentially random guessing at 25% would be better)
- The prompt doesn't include answer choices (A, B, C, D)
- The model generates free-form text instead of selecting A/B/C/D
- No constrained generation to force single-letter output

**What MRAG-Bench Actually Requires:**
```
Question: What anatomical structure is shown from this angle?
A) Left ventricle
B) Right atrium
C) Aortic arch
D) Pulmonary artery

Expected Output: C
Current Output: "Based on the medical images provided, the anatomical structure appears to be..."
```

---

## 🎯 Solution: 5-Step Improvement Plan

### **Phase 1: Add Multiple Choice Support (CRITICAL)**

#### 1.1 Update MultimodalContext to Include Choices

**File:** `src/generation/interface.py`

**Current:**
```python
@dataclass
class MultimodalContext:
    question: str
    images: List[Image.Image]
    image_paths: List[str] = None
    context_metadata: Dict[str, Any] = None
```

**Change to:**
```python
@dataclass
class MultimodalContext:
    question: str
    images: List[Image.Image]
    choices: Dict[str, str] = None  # NEW: {'A': 'text', 'B': 'text', ...}
    image_paths: List[str] = None
    context_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []
        if self.context_metadata is None:
            self.context_metadata = {}
        if self.choices is None:
            self.choices = {}
```

#### 1.2 Update Pipeline to Pass Choices

**File:** `src/pipeline.py`

**Find:** Line ~403 where `MultimodalContext` is created

**Current:**
```python
context = MultimodalContext(
    question=question,
    images=retrieved_images,
    image_paths=image_paths,
    context_metadata={
        "question_id": question_id,
        ...
    }
)
```

**Change to:**
```python
# Extract choices from ground_truth if it's a dict/json
choices = {}
if ground_truth and isinstance(ground_truth, str):
    # ground_truth might be just "A" or might contain choices
    # We need to get choices from the dataset sample
    pass  # Will be populated from dataset

context = MultimodalContext(
    question=question,
    images=retrieved_images,
    choices=choices,  # NEW
    image_paths=image_paths,
    context_metadata={
        "question_id": question_id,
        "ground_truth": ground_truth,
        "choices": choices,  # NEW
        ...
    }
)
```

#### 1.3 Update Evaluator to Extract Choices from Dataset

**File:** `src/evaluation/evaluator.py`

**Find:** Where samples are passed to pipeline (around line 193)

**Add before pipeline call:**
```python
# Extract choices from sample if available
choices = {}
if hasattr(sample, 'choices') and sample.choices:
    choices = sample.choices
elif hasattr(sample, 'metadata') and 'choices' in sample.metadata:
    choices = sample.metadata['choices']

# Pass choices to pipeline
result = self.pipeline.process_query(
    question=sample.question,
    question_id=sample.question_id,
    ground_truth=sample.ground_truth,
    choices=choices  # NEW parameter
)
```

#### 1.4 Fix Prompt Construction (MOST CRITICAL)

**File:** `src/generation/llava_pipeline.py`

**Replace the entire `construct_prompt` method (lines 258-294):**

```python
def construct_prompt(self, context: MultimodalContext) -> str:
    """
    Construct optimized prompt for multiple-choice medical image question answering.

    Args:
        context: MultimodalContext with question, images, and choices

    Returns:
        Formatted prompt string optimized for multiple-choice format
    """
    # Add image tokens for each image (required by LLaVA)
    image_tokens = "<image>\n" * len(context.images)

    # Format the question
    question = context.question.strip()
    if not question.endswith(('?', '.', '!')):
        question += "?"

    # Format choices if available
    choices_text = ""
    if context.choices and len(context.choices) > 0:
        choices_text = "\n\nChoices:\n"
        for letter in sorted(context.choices.keys()):
            choices_text += f"{letter}) {context.choices[letter]}\n"

    # Multiple-choice specific system prompt
    system_prompt = (
        "You are a medical AI assistant specialized in analyzing medical images. "
        "Answer the following multiple-choice question by selecting the single correct option. "
        "Examine the medical images carefully and choose the best answer.\n\n"
        "IMPORTANT: Respond with ONLY the letter (A, B, C, or D) of the correct answer. "
        "Do not provide any explanation or additional text."
    )

    # Construct final prompt
    prompt = f"{image_tokens}{system_prompt}\n\n{question}{choices_text}\n\nAnswer:"

    return prompt
```

#### 1.5 Add Answer Extraction and Validation

**File:** `src/generation/llava_pipeline.py`

**Add new method after `construct_prompt`:**

```python
def _extract_answer_choice(self, response: str) -> str:
    """
    Extract the answer choice (A, B, C, D) from model response.

    Args:
        response: Raw model output

    Returns:
        Single letter (A, B, C, or D) or empty string if not found
    """
    import re

    # Remove whitespace
    response = response.strip().upper()

    # Check if response is already just a single letter
    if response in ['A', 'B', 'C', 'D']:
        return response

    # Try to find first occurrence of A, B, C, or D
    # Pattern: Look for standalone letter (not part of word)
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)

    # Check first character if it's a valid choice
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]

    # Default to empty if no valid choice found
    return ""
```

**Update the generation code (around line 206-220):**

**Find:**
```python
response = self.processor.tokenizer.decode(
    response_ids,
    skip_special_tokens=True
).strip()

# Post-process response for medical domain
response = self._post_process_response(response)
```

**Change to:**
```python
response = self.processor.tokenizer.decode(
    response_ids,
    skip_special_tokens=True
).strip()

# Extract answer choice for multiple-choice questions
response = self._extract_answer_choice(response)

# If no valid choice extracted, default to "A"
if not response:
    logger.warning("Could not extract valid answer choice, defaulting to 'A'")
    response = "A"
```

---

### **Phase 2: Optimize Generation Parameters**

#### 2.1 Constrain Generation Length

**File:** `src/generation/llava_pipeline.py`

**Find:** Where `generation_kwargs` are set (around line 130-140)

**Add:**
```python
self.generation_kwargs.update({
    'max_new_tokens': 5,  # Force very short output (just need "A" or "B" etc)
    'min_new_tokens': 1,  # At least generate something
    'num_beams': 1,  # Greedy decoding for deterministic output
    'do_sample': False,  # Disable sampling for consistency
    'temperature': None,  # Not used when do_sample=False
    'repetition_penalty': 1.0,
    'length_penalty': 0.0,
})
```

---

### **Phase 3: Improve Retrieval Quality**

#### 3.1 Increase Top-K Retrieved Images

**File:** `config/mrag_bench.yaml`

**Current:**
```yaml
retrieval:
  top_k: 5
```

**Change to:**
```yaml
retrieval:
  top_k: 10  # Retrieve more candidates for better context
```

#### 3.2 Add Retrieval Reranking (Optional Advanced)

Consider adding a reranking step to select the most relevant images from top-10.

---

### **Phase 4: Dataset Format Verification**

#### 4.1 Verify Dataset Has Choices

**Run this diagnostic:**

```bash
source mrag-bench-env/bin/activate
python -c "
from src.dataset import MRAGDataset
from src.config import MRAGConfig

config = MRAGConfig()
dataset = MRAGDataset(config.dataset)
samples = dataset.get_samples_by_scenario('angle')

print('Sample structure:')
sample = samples[0]
print(f'Question: {sample.question}')
print(f'Ground Truth: {sample.ground_truth}')
print(f'Has choices attr: {hasattr(sample, \"choices\")}')
if hasattr(sample, 'choices'):
    print(f'Choices: {sample.choices}')
if hasattr(sample, 'metadata'):
    print(f'Metadata keys: {list(sample.metadata.keys())}')
"
```

If choices aren't in the Sample dataclass, you need to add them during dataset loading.

#### 4.2 Update Dataset Sample Class

**File:** `src/dataset/interface.py` or `src/dataset/mrag_dataset.py`

**Find the Sample dataclass and add choices field:**

```python
@dataclass
class Sample:
    question_id: str
    question: str
    ground_truth: str
    choices: Dict[str, str]  # NEW: {'A': 'text', 'B': 'text', 'C': 'text', 'D': 'text'}
    image_path: str = ""
    scenario: str = ""
    metadata: Dict[str, Any] = None
```

#### 4.3 Update Dataset Loader to Extract Choices

**File:** `src/dataset/mrag_dataset.py`

**In the method that loads samples, extract choices from the dataset:**

```python
# When creating Sample objects, extract choices
choices = {
    'A': item.get('A', ''),
    'B': item.get('B', ''),
    'C': item.get('C', ''),
    'D': item.get('D', '')
}

sample = Sample(
    question_id=question_id,
    question=question,
    ground_truth=ground_truth,
    choices=choices,  # NEW
    image_path=image_path,
    scenario=scenario,
    metadata=metadata
)
```

---

### **Phase 5: Prompt Engineering Refinements**

#### 5.1 Test Different Prompt Templates

Create multiple prompt variants and A/B test them:

**Variant 1: Extremely Concise**
```python
prompt = f"{image_tokens}Question: {question}\n\n{choices_text}\nAnswer (A/B/C/D only):"
```

**Variant 2: Chain-of-Thought (might help reasoning)**
```python
prompt = f"{image_tokens}Analyze the medical images and answer this question:\n\n{question}\n\n{choices_text}\n\nThink step by step, then provide ONLY the letter of your final answer (A, B, C, or D):\nAnswer:"
```

**Variant 3: Few-Shot (add examples)**
```python
examples = """
Example 1:
Question: What view is shown?
A) Frontal  B) Lateral  C) Oblique  D) Axial
Answer: A

Example 2:
Question: Which structure is visible?
A) Heart  B) Lung  C) Liver  D) Kidney
Answer: B

Now answer this question:
"""
prompt = f"{image_tokens}{examples}{question}\n\n{choices_text}\nAnswer:"
```

---

## 📊 Expected Accuracy Improvements

| Phase | Expected Improvement | Cumulative Accuracy |
|-------|---------------------|---------------------|
| Baseline | - | 1.5% |
| Phase 1: Multiple Choice Format | +20-30% | 21.5-31.5% |
| Phase 2: Generation Constraints | +10-15% | 31.5-46.5% |
| Phase 3: Better Retrieval | +5-10% | 36.5-56.5% |
| Phase 4: Dataset Fixes | +2-5% | 38.5-61.5% |
| Phase 5: Prompt Engineering | +10-15% | 48.5-76.5% |

**Target Range:** 53-59% - Should be achievable with Phases 1-3

---

## 🚀 Implementation Priority

### **IMMEDIATE (Do First):**
1. ✅ **Phase 1.4** - Fix prompt to include choices and force letter-only output
2. ✅ **Phase 1.5** - Add answer extraction logic
3. ✅ **Phase 2.1** - Constrain generation to max 5 tokens
4. ✅ **Phase 4.1** - Verify dataset has choices

### **HIGH PRIORITY (Do Next):**
5. **Phase 1.1-1.3** - Update dataflow to pass choices through pipeline
6. **Phase 4.2-4.3** - Ensure choices are in Sample objects
7. **Phase 3.1** - Increase retrieval top-k

### **MEDIUM PRIORITY (Optimization):**
8. **Phase 5** - Test different prompt variants
9. **Phase 3.2** - Add retrieval reranking

---

## 🧪 Testing Strategy

### Test After Each Phase:

```bash
# Quick test (10 samples) after each change
source mrag-bench-env/bin/activate
python run_sprint10_final_validation.py --quick-test --max-samples 10

# Full test when phases 1-3 complete
python run_sprint10_final_validation.py --num-runs 3
```

### Debug Individual Questions:

```python
# Test prompt generation
from src.generation import LLaVAGenerationPipeline, MultimodalContext
from PIL import Image

context = MultimodalContext(
    question="What anatomical structure is shown?",
    images=[Image.open("test.jpg")],
    choices={
        'A': 'Left ventricle',
        'B': 'Right atrium',
        'C': 'Aortic arch',
        'D': 'Pulmonary artery'
    }
)

pipeline = LLaVAGenerationPipeline(config)
pipeline.load_model()
prompt = pipeline.construct_prompt(context)
print("Generated prompt:")
print(prompt)
```

---

## 📝 Implementation Checklist

- [ ] Phase 1.1: Update MultimodalContext with choices field
- [ ] Phase 1.2: Update pipeline to pass choices
- [ ] Phase 1.3: Update evaluator to extract choices from dataset
- [ ] Phase 1.4: Fix prompt construction to include choices
- [ ] Phase 1.5: Add answer extraction logic
- [ ] Phase 2.1: Constrain generation parameters
- [ ] Phase 3.1: Increase top-k to 10
- [ ] Phase 4.1: Verify dataset format
- [ ] Phase 4.2: Update Sample dataclass
- [ ] Phase 4.3: Update dataset loader
- [ ] Phase 5.1: Test prompt variants
- [ ] Run full evaluation
- [ ] Verify 53-59% accuracy achieved

---

## 🎯 Success Criteria

- ✅ Accuracy reaches 53-59% range
- ✅ All 4 scenarios within ±5% of target
- ✅ Generated answers are valid choices (A/B/C/D)
- ✅ Processing time remains <2s per question
- ✅ Memory usage stays <6GB VRAM

---

## 💡 Additional Optimizations (If Needed)

If accuracy is still below target after Phase 1-5:

1. **Fine-tune LLaVA** on medical multiple-choice data
2. **Use larger model** (LLaVA-1.5-13B instead of 7B)
3. **Ensemble methods** (multiple model outputs + voting)
4. **Advanced retrieval** (hybrid dense+sparse retrieval)
5. **Image preprocessing** (enhance contrast, normalize, etc.)

---

**Next Steps:** Start with Phase 1.4 (prompt fix) and Phase 1.5 (answer extraction) - these are the quickest wins that should immediately improve accuracy from 1.5% to 20-30%+.
