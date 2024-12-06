import streamlit as st
import requests
from PIL import Image
import io

st.title("Image Classification & Text-to-Image Generator")
tab1, tab2 = st.tabs(["Image Classification", "Text-to-Image Generator"])

# Tab 1: Image Classification
with tab1:
    st.header("Image Classification")

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    # Classification labels input
    labels = st.text_input("Enter Classification Labels (comma-separated)", value="고양이,강아지,자동차,비행기")

    if st.button("Classify Image"):
        if uploaded_image is None:
            st.error("Please upload an image to classify.")
        elif not labels.strip():
            st.error("Please enter valid classification labels.")
        else:
            # Perform classification
            with st.spinner("Classifying image..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/image-to-text",
                        files={"file": uploaded_image},
                        data={"labels": labels}
                    )
                    if response.status_code == 200:
                        st.success("Classification successful!")
                        results = response.json()["results"]

                        # Extract max label and probability
                        max_label = max(results, key=results.get)
                        max_prob = results[max_label]

                        st.write(f"**{max_label}**일 확률은 {max_prob:.2f}%입니다.")
                    else:
                        st.error(f"Error: {response.json()['error']}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")

# Tab 2: Text-to-Image Generator
with tab2:
    st.header("Text-to-Image Generator")

    # Prompt input
    prompt = st.text_input("Enter Text Prompt")

    if st.button("Generate Image"):
        if not prompt.strip():
            st.error("Please enter a valid prompt.")
        else:
            # Perform text-to-image generation
            with st.spinner("Generating image..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/text-to-image",
                        data={"prompt": prompt}
                    )
                    if response.status_code == 200:
                        st.success("Image generated successfully!")
                        image = Image.open(io.BytesIO(response.content))
                        st.image(image, caption="Generated Image", use_container_width=True)
                    else:
                        st.error(f"Error: {response.json()['error']}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by FastAPI & Streamlit")