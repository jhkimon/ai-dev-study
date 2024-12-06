from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
from app.models import CLIPClassifier, TextToImageGenerator, ImageComparator
from PIL import Image
import io
import torch

app = FastAPI()
clip_classifier = CLIPClassifier()
text_to_image_generator = TextToImageGenerator()
image_comparator = ImageComparator(clip_classifier)

@app.post("/text-to-image")
async def text_to_image(prompt: str = Form(...)):
    """
    Generate an image from a text prompt using a text-to-image model (Diffusion).

    Args:
        prompt (str): The text prompt describing the desired image.

    Returns:
        StreamingResponse: A PNG image generated from the provided prompt.
    """
    try:
        # Generate the image
        generated_image = text_to_image_generator.generate_image(prompt)

        # Save the image to a BytesIO stream
        image_stream = io.BytesIO()
        generated_image.save(image_stream, format="PNG")
        image_stream.seek(0)

        # Return the image as a streaming response
        return StreamingResponse(image_stream, media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/image-to-text")
async def classify_image(
    file: UploadFile = File(...), 
    labels: str = Form("고양이,강아지,자동차,비행기")
):
    """
    Classify an uploaded image based on provided labels using a CLIP model.

    Args:
        file (UploadFile): The uploaded image file to be classified.
        labels (str): A comma-separated string of labels for classification.

    Returns:
        JSONResponse: A JSON object containing the labels and their corresponding probabilities.
    """
    try:
        # Label split
        labels_list = [label.strip() for label in labels.split(",") if label.strip()]
        print("Labels List:", labels_list)
        if not labels_list:
            return JSONResponse(content={"error": "Labels list is empty or invalid."}, status_code=400)

        # Read the uploaded file and process it as an image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print("Image size:", image.size)
        print("Image mode:", image.mode)

        # Perform classification
        results = clip_classifier.classify(image=image, labels=labels_list)
        
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    """
    Check the health status of the server.

    Returns:
        dict: A dictionary indicating the server's health status and message.
    """
    return {"status": "ok", "message": "Server is running smoothly."}

@app.post("/compare-images")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compares two images using the cosine similarity.

    Args:
        file1 (UploadFile): The first image file.
        file2 (UploadFile): The second image file.

    Returns:
        JSONResponse: A JSON object with the similarity score.
    """
    try:
        # Load and preprocess the first image
        image_data1 = await file1.read()
        image1 = Image.open(io.BytesIO(image_data1)).convert("RGB")

        # Load and preprocess the second image
        image_data2 = await file2.read()
        image2 = Image.open(io.BytesIO(image_data2)).convert("RGB")

        # Use the ImageComparator to calculate similarity
        similarity_score = image_comparator.compare(image1, image2)

        return JSONResponse(content={"similarity_score": similarity_score})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)