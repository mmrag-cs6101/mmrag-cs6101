# MRAG-Bench Reproduction System - API Documentation

**Version:** 1.0
**Date:** October 4, 2025
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration API](#configuration-api)
3. [Dataset API](#dataset-api)
4. [Retrieval API](#retrieval-api)
5. [Generation API](#generation-api)
6. [Pipeline API](#pipeline-api)
7. [Evaluation API](#evaluation-api)
8. [Utilities API](#utilities-api)
9. [Examples and Usage Patterns](#examples-and-usage-patterns)

---

## Overview

The MRAG-Bench system provides a comprehensive Python API for multimodal retrieval-augmented generation. This document covers all public APIs, data structures, and usage patterns.

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```python
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline

# Load configuration
config = MRAGConfig()

# Initialize pipeline
pipeline = MRAGPipeline(config)

# Process query
result = pipeline.process_query(
    question="What is the difference between these MRI scans?"
)

print(f"Answer: {result.generated_answer}")
```

---

## Configuration API

### MRAGConfig

**Module:** `src.config`

Complete system configuration with validation and serialization support.

#### Class Definition

```python
@dataclass
class MRAGConfig:
    """Complete MRAG-Bench system configuration."""
    model: ModelConfig
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    performance: PerformanceConfig
```

#### Subconfigurations

##### ModelConfig

```python
@dataclass
class ModelConfig:
    """Model configuration parameters."""
    vlm_name: str = "llava-hf/llava-1.5-7b-hf"
    retriever_name: str = "openai/clip-vit-base-patch32"
    quantization: str = "4bit"  # "4bit", "8bit", or "none"
    max_memory_gb: float = 14.0
    device: str = "cuda"
    torch_dtype: str = "float16"
```

##### DatasetConfig

```python
@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""
    data_path: str = "data/mrag_bench"
    batch_size: int = 4
    image_size: tuple = (224, 224)
    cache_embeddings: bool = True
    embedding_cache_path: str = "data/embeddings"
```

##### RetrievalConfig

```python
@dataclass
class RetrievalConfig:
    """Retrieval configuration parameters."""
    embedding_dim: int = 512
    top_k: int = 5
    similarity_threshold: float = 0.0
    batch_size: int = 16
    faiss_index_type: str = "IVF"
    index_cache_path: str = "data/embeddings/faiss_index.bin"
```

##### GenerationConfig

```python
@dataclass
class GenerationConfig:
    """Generation configuration parameters."""
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
```

##### PerformanceConfig

```python
@dataclass
class PerformanceConfig:
    """Performance and resource constraints."""
    memory_limit_gb: float = 16.0
    memory_buffer_gb: float = 1.0
    retrieval_timeout: float = 5.0
    generation_timeout: float = 25.0
    total_pipeline_timeout: float = 30.0
    enable_memory_monitoring: bool = True
```

#### Methods

##### from_yaml()

Load configuration from YAML file.

```python
@classmethod
def from_yaml(cls, filepath: str) -> 'MRAGConfig':
    """Load configuration from YAML file."""
```

**Example:**

```python
config = MRAGConfig.from_yaml("config/mrag_bench.yaml")
```

##### from_dict()

Create configuration from dictionary.

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'MRAGConfig':
    """Create configuration from dictionary."""
```

**Example:**

```python
config_dict = {
    "model": {"vlm_name": "llava-hf/llava-1.5-7b-hf"},
    "retrieval": {"top_k": 10}
}
config = MRAGConfig.from_dict(config_dict)
```

##### validate()

Validate configuration parameters.

```python
def validate(self) -> bool:
    """Validate configuration parameters."""
```

**Example:**

```python
config = MRAGConfig()
if config.validate():
    print("Configuration is valid")
```

##### to_dict()

Convert configuration to dictionary.

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
```

##### save()

Save configuration to YAML file.

```python
def save(self, filepath: str) -> None:
    """Save configuration to YAML file."""
```

**Example:**

```python
config = MRAGConfig()
config.save("config/my_config.yaml")
```

---

## Dataset API

### MRAGDataset

**Module:** `src.dataset`

MRAG-Bench dataset loader with perspective change scenario filtering.

#### Class Definition

```python
class MRAGDataset:
    """
    MRAG-Bench dataset with streaming support and scenario filtering.

    Features:
    - Memory-efficient dataset loading
    - Perspective change scenario filtering
    - Image preprocessing and validation
    - Batch processing support
    """
```

#### Constructor

```python
def __init__(
    self,
    data_path: str = "data/mrag_bench",
    batch_size: int = 4,
    image_size: Tuple[int, int] = (224, 224)
):
    """
    Initialize MRAG dataset.

    Args:
        data_path: Path to MRAG-Bench dataset directory
        batch_size: Batch size for streaming
        image_size: Target image size for preprocessing
    """
```

**Example:**

```python
from src.dataset import MRAGDataset

dataset = MRAGDataset(
    data_path="data/mrag_bench",
    batch_size=8,
    image_size=(224, 224)
)
```

#### Methods

##### get_sample()

Get single sample by index.

```python
def get_sample(self, index: int) -> Sample:
    """Get sample by index."""
```

**Example:**

```python
sample = dataset.get_sample(0)
print(f"Question: {sample.question}")
print(f"Image paths: {sample.image_paths}")
print(f"Answer: {sample.answer}")
```

##### filter_by_scenario()

Filter samples by perspective change scenario.

```python
def filter_by_scenario(
    self,
    scenario_type: str
) -> List[Sample]:
    """
    Filter samples by scenario type.

    Args:
        scenario_type: One of ["angle", "partial", "scope", "occlusion"]

    Returns:
        List of samples matching the scenario
    """
```

**Example:**

```python
angle_samples = dataset.filter_by_scenario("angle")
print(f"Found {len(angle_samples)} angle change samples")
```

##### get_retrieval_corpus()

Get all image paths for retrieval corpus.

```python
def get_retrieval_corpus(self) -> List[str]:
    """Get all image paths for retrieval corpus."""
```

**Example:**

```python
corpus_paths = dataset.get_retrieval_corpus()
print(f"Corpus contains {len(corpus_paths)} images")
```

##### get_available_scenarios()

Get list of available perspective change scenarios.

```python
def get_available_scenarios(self) -> List[str]:
    """Get list of available scenarios."""
```

**Example:**

```python
scenarios = dataset.get_available_scenarios()
print(f"Available scenarios: {scenarios}")
# Output: ['angle', 'partial', 'scope', 'occlusion']
```

##### validate_dataset()

Validate dataset integrity and structure.

```python
def validate_dataset(self) -> Dict[str, Any]:
    """
    Validate dataset integrity.

    Returns:
        Validation results dictionary with status and details
    """
```

**Example:**

```python
validation = dataset.validate_dataset()
if validation["status"] == "success":
    print(f"Dataset valid: {validation['total_samples']} samples")
else:
    print(f"Dataset errors: {validation['errors']}")
```

#### Data Structures

##### Sample

```python
@dataclass
class Sample:
    """MRAG-Bench sample with metadata."""
    question_id: str
    question: str
    answer: str
    image_paths: List[str]
    scenario_type: str
    metadata: Dict[str, Any]
```

---

## Retrieval API

### CLIPRetriever

**Module:** `src.retrieval`

CLIP ViT-B/32 based image retrieval with FAISS indexing.

#### Class Definition

```python
class CLIPRetriever(RetrievalPipeline):
    """
    CLIP-based image retrieval pipeline.

    Features:
    - CLIP ViT-B/32 model integration
    - Batch embedding generation
    - FAISS vector indexing
    - Top-k similarity search
    - Persistent index caching
    """
```

#### Constructor

```python
def __init__(self, config: RetrievalConfig):
    """
    Initialize CLIP retriever.

    Args:
        config: Retrieval configuration parameters
    """
```

**Example:**

```python
from src.retrieval import CLIPRetriever, RetrievalConfig
from src.config import MRAGConfig

config = MRAGConfig()
retrieval_config = RetrievalConfig(
    model_name=config.model.retriever_name,
    embedding_dim=config.retrieval.embedding_dim,
    top_k=config.retrieval.top_k,
    device=config.model.device
)

retriever = CLIPRetriever(retrieval_config)
```

#### Methods

##### load_model()

Load CLIP model with memory optimization.

```python
def load_model(self) -> None:
    """Load CLIP model with memory optimization."""
```

**Example:**

```python
retriever.load_model()
```

##### encode_images()

Generate embeddings for images.

```python
def encode_images(
    self,
    images: List[Image.Image]
) -> np.ndarray:
    """
    Generate embeddings for batch of images.

    Args:
        images: List of PIL images

    Returns:
        Numpy array of embeddings (batch_size, embedding_dim)
    """
```

**Example:**

```python
from PIL import Image

images = [Image.open(path) for path in image_paths[:10]]
embeddings = retriever.encode_images(images)
print(f"Generated embeddings: {embeddings.shape}")  # (10, 512)
```

##### encode_text()

Generate embedding for text query.

```python
def encode_text(self, text: str) -> np.ndarray:
    """
    Generate embedding for text query.

    Args:
        text: Query text

    Returns:
        Numpy array of embedding (1, embedding_dim)
    """
```

**Example:**

```python
query = "Show MRI scan with angle change"
query_embedding = retriever.encode_text(query)
print(f"Query embedding shape: {query_embedding.shape}")  # (1, 512)
```

##### build_index()

Build FAISS index for image corpus.

```python
def build_index(
    self,
    image_paths: List[str],
    force_rebuild: bool = False
) -> None:
    """
    Build FAISS index for image corpus.

    Args:
        image_paths: List of image file paths
        force_rebuild: Force rebuilding even if cache exists
    """
```

**Example:**

```python
corpus_paths = dataset.get_retrieval_corpus()
retriever.build_index(corpus_paths)
```

##### retrieve()

Retrieve top-k similar images for query.

```python
def retrieve(
    self,
    query: str,
    top_k: Optional[int] = None
) -> List[RetrievalResult]:
    """
    Retrieve top-k similar images for query.

    Args:
        query: Query text
        top_k: Number of images to retrieve (default: from config)

    Returns:
        List of retrieval results with image paths and scores
    """
```

**Example:**

```python
query = "What is the difference between these MRI scans?"
results = retriever.retrieve(query, top_k=5)

for i, result in enumerate(results):
    print(f"{i+1}. {result.image_path} (score: {result.score:.3f})")
```

#### Data Structures

##### RetrievalResult

```python
@dataclass
class RetrievalResult:
    """Single retrieval result."""
    image_path: str
    score: float
    rank: int
    metadata: Dict[str, Any]
```

---

## Generation API

### LLaVAGenerationPipeline

**Module:** `src.generation`

LLaVA-1.5-7B generation pipeline with 4-bit quantization.

#### Class Definition

```python
class LLaVAGenerationPipeline(GenerationPipeline):
    """
    LLaVA-based answer generation pipeline.

    Features:
    - LLaVA-1.5-7B with 4-bit quantization
    - Multimodal prompt construction
    - Medical domain-specific generation
    - Memory-optimized inference
    - Comprehensive error handling
    """
```

#### Constructor

```python
def __init__(self, config: GenerationConfig):
    """
    Initialize LLaVA generation pipeline.

    Args:
        config: Generation configuration parameters
    """
```

**Example:**

```python
from src.generation import LLaVAGenerationPipeline, GenerationConfig
from src.config import MRAGConfig

config = MRAGConfig()
gen_config = GenerationConfig(
    model_name=config.model.vlm_name,
    max_length=config.generation.max_length,
    temperature=config.generation.temperature,
    quantization=config.model.quantization,
    device=config.model.device
)

generator = LLaVAGenerationPipeline(gen_config)
```

#### Methods

##### load_model()

Load LLaVA model with quantization.

```python
def load_model(self) -> None:
    """Load LLaVA model with 4-bit quantization."""
```

**Example:**

```python
generator.load_model()
```

##### generate_answer()

Generate answer for multimodal context.

```python
def generate_answer(
    self,
    context: MultimodalContext
) -> GenerationResult:
    """
    Generate answer for multimodal context.

    Args:
        context: Multimodal context with question and images

    Returns:
        Generation result with answer and metadata
    """
```

**Example:**

```python
from src.generation import MultimodalContext
from PIL import Image

# Load images
images = [Image.open(path) for path in retrieved_image_paths]

# Create context
context = MultimodalContext(
    question="What is the difference between these scans?",
    images=images,
    metadata={}
)

# Generate answer
result = generator.generate_answer(context)
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Generation time: {result.generation_time:.2f}s")
```

##### unload_model()

Unload model from memory.

```python
def unload_model(self) -> None:
    """Unload model to free memory."""
```

**Example:**

```python
generator.unload_model()
```

#### Data Structures

##### MultimodalContext

```python
@dataclass
class MultimodalContext:
    """Multimodal context for generation."""
    question: str
    images: List[Image.Image]
    metadata: Dict[str, Any]
    retrieved_image_paths: Optional[List[str]] = None
```

##### GenerationResult

```python
@dataclass
class GenerationResult:
    """Generation result with metadata."""
    answer: str
    confidence_score: float
    generation_time: float
    input_tokens: int
    output_tokens: int
    memory_usage: Dict[str, float]
    metadata: Dict[str, Any]
```

---

## Pipeline API

### MRAGPipeline

**Module:** `src.pipeline`

End-to-end MRAG pipeline orchestrating retrieval and generation.

#### Class Definition

```python
class MRAGPipeline:
    """
    Complete MRAG pipeline.

    Features:
    - Sequential model loading for memory management
    - Dynamic memory optimization
    - Error handling and recovery
    - Performance monitoring
    - Comprehensive logging
    """
```

#### Constructor

```python
def __init__(self, config: MRAGConfig):
    """
    Initialize MRAG pipeline.

    Args:
        config: Complete MRAG system configuration
    """
```

**Example:**

```python
from src.pipeline import MRAGPipeline
from src.config import MRAGConfig

config = MRAGConfig()
pipeline = MRAGPipeline(config)
```

#### Methods

##### initialize_dataset()

Initialize dataset component.

```python
def initialize_dataset(self) -> None:
    """Initialize dataset component."""
```

**Example:**

```python
pipeline.initialize_dataset()
```

##### load_retriever()

Load CLIP retrieval component.

```python
def load_retriever(self) -> None:
    """Load CLIP retriever and build index."""
```

**Example:**

```python
pipeline.load_retriever()
```

##### load_generator()

Load LLaVA generation component.

```python
def load_generator(self) -> None:
    """Load LLaVA generator with quantization."""
```

**Example:**

```python
pipeline.load_generator()
```

##### process_query()

Process complete query through pipeline.

```python
def process_query(
    self,
    question: str,
    question_id: Optional[str] = None,
    scenario_type: Optional[str] = None
) -> PipelineResult:
    """
    Process query through complete pipeline.

    Args:
        question: Query text
        question_id: Optional question ID for tracking
        scenario_type: Optional scenario type for filtering

    Returns:
        Complete pipeline result with retrieval and generation
    """
```

**Example:**

```python
result = pipeline.process_query(
    question="What is the difference between these MRI scans?",
    question_id="q001",
    scenario_type="angle"
)

print(f"Question: {result.question}")
print(f"Retrieved {len(result.retrieved_images)} images")
print(f"Answer: {result.generated_answer}")
print(f"Total time: {result.total_time:.2f}s")
print(f"  Retrieval: {result.retrieval_time:.2f}s")
print(f"  Generation: {result.generation_time:.2f}s")
```

##### get_pipeline_stats()

Get pipeline performance statistics.

```python
def get_pipeline_stats(self) -> Dict[str, Any]:
    """Get pipeline performance statistics."""
```

**Example:**

```python
stats = pipeline.get_pipeline_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['successful_queries']/stats['total_queries']:.1%}")
print(f"Avg time: {stats['avg_total_time']:.2f}s")
```

#### Data Structures

##### PipelineResult

```python
@dataclass
class PipelineResult:
    """Complete pipeline result."""
    question_id: str
    question: str
    retrieved_images: List[str]
    retrieval_scores: List[float]
    generated_answer: str
    confidence_score: float
    total_time: float
    retrieval_time: float
    generation_time: float
    memory_usage: Dict[str, float]
    metadata: Dict[str, Any]
    pipeline_stage_times: Dict[str, float]
    memory_usage_per_stage: Dict[str, Dict[str, float]]
    error_recovery_attempts: int
    optimization_triggers: List[str]
```

---

## Evaluation API

### MRAGBenchEvaluator

**Module:** `src.evaluation`

MRAG-Bench evaluation framework implementing comprehensive evaluation methodology.

#### Class Definition

```python
class MRAGBenchEvaluator:
    """
    MRAG-Bench evaluator.

    Features:
    - Perspective change scenario evaluation
    - Automated accuracy calculation
    - Performance metrics collection
    - Memory usage monitoring
    - Error analysis
    - Comprehensive reporting
    """
```

#### Constructor

```python
def __init__(
    self,
    config: MRAGConfig,
    output_dir: str = "evaluation_results"
):
    """
    Initialize MRAG-Bench evaluator.

    Args:
        config: Complete MRAG system configuration
        output_dir: Directory for results and reports
    """
```

**Example:**

```python
from src.evaluation import MRAGBenchEvaluator
from src.config import MRAGConfig

config = MRAGConfig()
evaluator = MRAGBenchEvaluator(
    config=config,
    output_dir="output/evaluation"
)
```

#### Methods

##### initialize_components()

Initialize dataset and pipeline components.

```python
def initialize_components(self) -> None:
    """Initialize evaluation components."""
```

**Example:**

```python
evaluator.initialize_components()
```

##### evaluate_scenario()

Evaluate specific perspective change scenario.

```python
def evaluate_scenario(
    self,
    scenario_type: str,
    max_samples: Optional[int] = None,
    use_cache: bool = True
) -> ScenarioMetrics:
    """
    Evaluate specific scenario.

    Args:
        scenario_type: One of ["angle", "partial", "scope", "occlusion"]
        max_samples: Maximum samples to evaluate (None = all)
        use_cache: Use cached results if available

    Returns:
        Scenario evaluation metrics
    """
```

**Example:**

```python
# Evaluate angle change scenario
metrics = evaluator.evaluate_scenario(
    scenario_type="angle",
    max_samples=100
)

print(f"Scenario: {metrics.scenario_type}")
print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"Total questions: {metrics.total_questions}")
print(f"Correct answers: {metrics.correct_answers}")
print(f"Avg processing time: {metrics.avg_processing_time:.2f}s")
```

##### evaluate_all_scenarios()

Evaluate all perspective change scenarios.

```python
def evaluate_all_scenarios(
    self,
    max_samples_per_scenario: Optional[int] = None
) -> EvaluationSession:
    """
    Evaluate all scenarios.

    Args:
        max_samples_per_scenario: Max samples per scenario

    Returns:
        Complete evaluation session results
    """
```

**Example:**

```python
session = evaluator.evaluate_all_scenarios(
    max_samples_per_scenario=200
)

print(f"Overall accuracy: {session.overall_accuracy:.1%}")
print(f"Total questions: {session.total_questions}")
print(f"\nPer-scenario results:")
for scenario, metrics in session.scenario_results.items():
    print(f"  {scenario.upper()}: {metrics.accuracy:.1%} ({metrics.total_questions} samples)")
```

##### save_results()

Save evaluation results to file.

```python
def save_results(
    self,
    session: EvaluationSession,
    filename: Optional[str] = None
) -> str:
    """
    Save evaluation results.

    Args:
        session: Evaluation session to save
        filename: Optional output filename

    Returns:
        Path to saved results file
    """
```

**Example:**

```python
results_path = evaluator.save_results(session)
print(f"Results saved to: {results_path}")
```

#### Data Structures

##### ScenarioMetrics

```python
@dataclass
class ScenarioMetrics:
    """Metrics for specific scenario."""
    scenario_type: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_processing_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    confidence_scores: List[float]
    error_count: int
    error_rate: float
```

##### EvaluationSession

```python
@dataclass
class EvaluationSession:
    """Complete evaluation session."""
    session_id: str
    timestamp: str
    config_summary: Dict[str, Any]
    scenario_results: Dict[str, ScenarioMetrics]
    overall_accuracy: float
    total_questions: int
    total_correct: int
    avg_processing_time: float
    memory_stats: Dict[str, float]
    error_analysis: Dict[str, Any]
```

---

## Utilities API

### MemoryManager

**Module:** `src.utils.memory_manager`

GPU memory management and monitoring utilities.

#### Class Definition

```python
class MemoryManager:
    """
    GPU memory management.

    Features:
    - Real-time memory monitoring
    - Automatic cleanup on threshold
    - Memory guards for operations
    - Emergency cleanup procedures
    """
```

#### Constructor

```python
def __init__(
    self,
    memory_limit_gb: float = 15.0,
    buffer_gb: float = 1.0
):
    """
    Initialize memory manager.

    Args:
        memory_limit_gb: Maximum allowed memory usage
        buffer_gb: Memory buffer for safety
    """
```

**Example:**

```python
from src.utils.memory_manager import MemoryManager

manager = MemoryManager(
    memory_limit_gb=15.0,
    buffer_gb=1.0
)
```

#### Methods

##### check_memory_availability()

Check if sufficient memory is available.

```python
def check_memory_availability(
    self,
    required_gb: float
) -> bool:
    """Check if required memory is available."""
```

**Example:**

```python
if manager.check_memory_availability(required_gb=5.0):
    # Proceed with operation
    pass
else:
    manager.clear_gpu_memory()
```

##### clear_gpu_memory()

Clear GPU memory caches.

```python
def clear_gpu_memory(self) -> None:
    """Clear GPU memory caches."""
```

**Example:**

```python
manager.clear_gpu_memory()
```

##### get_memory_stats()

Get current memory statistics.

```python
def get_memory_stats(self) -> MemoryStats:
    """Get current memory statistics."""
```

**Example:**

```python
stats = manager.get_memory_stats()
print(f"GPU Memory: {stats.gpu_allocated_gb:.1f}/{stats.gpu_total_gb:.1f}GB")
print(f"Utilization: {stats.gpu_utilization_percent():.1f}%")
```

##### memory_guard()

Context manager for memory-safe operations.

```python
@contextmanager
def memory_guard(self, operation_name: str):
    """Context manager for memory-safe operations."""
```

**Example:**

```python
with manager.memory_guard("Model loading"):
    model.load()
    # Automatic cleanup on exception
```

---

## Examples and Usage Patterns

### Example 1: Basic Query Processing

```python
from src.config import MRAGConfig
from src.pipeline import MRAGPipeline

# Initialize pipeline
config = MRAGConfig()
pipeline = MRAGPipeline(config)

# Process query
result = pipeline.process_query(
    question="What anatomical structure is visible in this scan?"
)

print(f"Answer: {result.generated_answer}")
```

### Example 2: Custom Configuration

```python
from src.config import MRAGConfig, GenerationConfig

# Create custom configuration
config = MRAGConfig()

# Modify generation parameters
config.generation.temperature = 0.5  # More deterministic
config.generation.max_length = 256   # Shorter responses

# Modify retrieval parameters
config.retrieval.top_k = 10  # Retrieve more images

# Use custom configuration
pipeline = MRAGPipeline(config)
```

### Example 3: Batch Evaluation

```python
from src.evaluation import MRAGBenchEvaluator
from src.config import MRAGConfig

# Initialize evaluator
config = MRAGConfig()
evaluator = MRAGBenchEvaluator(config, output_dir="output/eval")

# Evaluate all scenarios
session = evaluator.evaluate_all_scenarios(max_samples_per_scenario=100)

# Print results
print(f"Overall Accuracy: {session.overall_accuracy:.1%}")
for scenario, metrics in session.scenario_results.items():
    print(f"{scenario}: {metrics.accuracy:.1%}")
```

### Example 4: Custom Retrieval

```python
from src.retrieval import CLIPRetriever, RetrievalConfig
from src.dataset import MRAGDataset
from PIL import Image

# Initialize components
dataset = MRAGDataset()
config = RetrievalConfig()
retriever = CLIPRetriever(config)

# Build index
corpus = dataset.get_retrieval_corpus()
retriever.build_index(corpus)

# Retrieve images
query = "Show MRI with angle change"
results = retriever.retrieve(query, top_k=5)

# Display results
for i, result in enumerate(results):
    print(f"{i+1}. {result.image_path} (score: {result.score:.3f})")
```

### Example 5: Memory Monitoring

```python
from src.utils.memory_manager import MemoryManager
from src.pipeline import MRAGPipeline
from src.config import MRAGConfig

# Initialize with memory monitoring
manager = MemoryManager(memory_limit_gb=15.0)
config = MRAGConfig()
pipeline = MRAGPipeline(config)

# Monitor memory during processing
stats_before = manager.get_memory_stats()
result = pipeline.process_query("Sample question")
stats_after = manager.get_memory_stats()

print(f"Memory used: {stats_after.gpu_allocated_gb - stats_before.gpu_allocated_gb:.2f}GB")
```

### Example 6: Error Handling

```python
from src.pipeline import MRAGPipeline
from src.utils.error_handling import MRAGError
from src.config import MRAGConfig

config = MRAGConfig()
pipeline = MRAGPipeline(config)

try:
    result = pipeline.process_query("Complex medical query")
    print(f"Success: {result.generated_answer}")
except MRAGError as e:
    print(f"MRAG Error: {e}")
    print(f"Category: {e.category}")
    print(f"Severity: {e.severity}")
```

### Example 7: Performance Benchmarking

```python
from src.evaluation import MRAGBenchEvaluator
from src.config import MRAGConfig
import time

config = MRAGConfig()
evaluator = MRAGBenchEvaluator(config)

# Benchmark single scenario
start_time = time.time()
metrics = evaluator.evaluate_scenario("angle", max_samples=100)
total_time = time.time() - start_time

print(f"Evaluated {metrics.total_questions} samples in {total_time:.2f}s")
print(f"Throughput: {metrics.total_questions/total_time:.2f} samples/sec")
print(f"Avg per-sample time: {metrics.avg_processing_time:.2f}s")
```

---

## Best Practices

### 1. Always Use Configuration Files

```python
# ✅ Good
config = MRAGConfig.from_yaml("config/production.yaml")

# ❌ Avoid
config = MRAGConfig()
config.model.vlm_name = "..."  # Manual modification
```

### 2. Initialize Components Explicitly

```python
# ✅ Good
pipeline = MRAGPipeline(config)
pipeline.initialize_dataset()
pipeline.load_retriever()
pipeline.load_generator()

# ❌ Avoid relying on implicit initialization
```

### 3. Use Memory Guards for Heavy Operations

```python
# ✅ Good
with memory_manager.memory_guard("Model loading"):
    model.load()

# ❌ Avoid unguarded operations
```

### 4. Handle Errors Gracefully

```python
# ✅ Good
try:
    result = pipeline.process_query(question)
except MRAGError as e:
    logger.error(f"Error: {e}")
    # Fallback or recovery

# ❌ Avoid silent failures
```

### 5. Save Evaluation Results

```python
# ✅ Good
session = evaluator.evaluate_all_scenarios()
evaluator.save_results(session, "results/final_eval.json")

# ❌ Don't discard results
```

---

**Document Version:** 1.0
**Last Updated:** October 4, 2025
**Maintained By:** MRAG-Bench Development Team

For additional support, see:
- [Implementation Guide](IMPLEMENTATION_GUIDE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Performance Analysis](PERFORMANCE_ANALYSIS.md)
