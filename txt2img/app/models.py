from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import List
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch



class TextToImageGenerator:
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-1"):
        """
        Initialize the Stable Diffusion model pipeline with optimized settings.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(device)

    def generate_image(
        self,
        prompt: str,
        image_size: tuple = (512, 512),
        num_inference_steps: int = 100,
        guidance_scale: float = 12.5, 
        negative_prompt: str = "blurry, distorted, cartoonish, unrealistic"  
    ) -> Image.Image:
        """
        Generate a image from the given text prompt.
        """
        # Generate the image using the pipeline with optimized parameters
        result = self.pipe(
            prompt=prompt,
            height=image_size[0],
            width=image_size[1],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )
        return result.images[0]  # Return the first image

       
class CLIPClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def classify(self, image: Image.Image, labels: list[str]) -> dict:
        """
        Generate a single image from the given text prompt.
        Reduce inference steps for faster generation.
        """
        
        # input 전처리
        try:
            inputs = self.processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            print("Processed inputs:", {k: v.shape for k, v in inputs.items()}) # item check
        except Exception as e:
            raise ValueError(f"Error during input processing: {e}")

        # Inference
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
        except Exception as e:
            raise ValueError(f"Error during inference: {e}") # 추론 중간의 오류 처리

        # 결과 반환
        try:
            results = {label: prob.item() * 100 for label, prob in zip(labels, probs[0])}
            return results
        except Exception as e:
            raise ValueError(f"Error during result formatting: {e}") # formatting 오류 처리
        
class ImageComparator:
    def __init__(self, classifier: CLIPClassifier):
        self.classifier = classifier

    def compare(self, image1: Image.Image, image2: Image.Image) -> float:
        try:
            # Preprocess the images
            inputs1 = self.classifier.processor(images=image1, return_tensors="pt")
            inputs2 = self.classifier.processor(images=image2, return_tensors="pt")

            # Get image embeddings
            with torch.no_grad():
                image_features1 = self.classifier.model.get_image_features(**inputs1)
                image_features2 = self.classifier.model.get_image_features(**inputs2)

            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(image_features1, image_features2) * 100
            return similarity.item()
        except Exception as e:
            raise ValueError(f"Error during comparison: {e}")