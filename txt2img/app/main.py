from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
from app.models import CLIPClassifier, TextToImageGenerator
from PIL import Image
import io

app = FastAPI()
clip_classifier = CLIPClassifier()
text_to_image_generator = TextToImageGenerator()

@app.post("/text-to-image")
async def text_to_image(prompt: str = Form(...)):
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
    return {"status": "ok", "message": "Server is running smoothly."}