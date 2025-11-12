"""Debug the actual prompt being sent to the model."""

from src.generation import LLaVAGenerationPipeline, GenerationConfig, MultimodalContext
from PIL import Image

# Create config
config = GenerationConfig(
    model_name="llava-hf/llava-1.5-7b-hf",
    max_length=50,
    temperature=0.3
)

# Create pipeline
pipeline = LLaVAGenerationPipeline(config)

# Create dummy context
images = [Image.new('RGB', (224, 224), (255, 255, 255))]  # White image
context = MultimodalContext(
    question="What animal is this?",
    images=images
)

# Generate prompt
prompt = pipeline.construct_prompt(context)

print("="*80)
print("GENERATED PROMPT:")
print("="*80)
print(prompt)
print("="*80)
print(f"Prompt length: {len(prompt)} characters")
print(f"Number of <image> tokens: {prompt.count('<image>')}")
