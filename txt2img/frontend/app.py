import streamlit as st
import requests
from PIL import Image
import io
import os
# from dotenv import load_dotenv

# .env
# load_dotenv()
SERVER_URL = "http://localhost:8000"

# 사진 불러오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

SAMPLE_IMAGE_1_PATH = os.path.join(ASSETS_DIR, "cha.jpg")
SAMPLE_IMAGE_2_PATH = os.path.join(ASSETS_DIR, "hanni.jpg")
SAMPLE_IMAGE_3_PATH = os.path.join(ASSETS_DIR, "cat.jpg")
st.title("Image Comparison")
tab1, tab2, tab3 = st.tabs(["닮은 동물 찾기", "닮은 사람 찾기", "오늘의 선물"])

# Tab 1: Image Classification
with tab1:
    st.header("나와 닮은 동물 찾기")

    # Upload image
    uploaded_image = st.file_uploader("이미지를 올려주세요.", type=["jpg", "jpeg", "png"])
    
    # Classification labels input
    labels = st.text_input("Enter Classification Labels (comma-separated)", value="고양이,강아지,여우,호랑이")

    if st.button("Classify Image"):
        if uploaded_image is None:
            st.error("이미지를 올려주세요.")
        elif not labels.strip():
            st.error("비교하고 싶은 대상을 입력해주세요.")
        else:
            # Perform classification
            with st.spinner("이미지 분류중..."):
                try:
                    response = requests.post(
                        
                        f"{SERVER_URL}/image-to-text",
                        files={"file": uploaded_image},
                        data={"labels": labels}
                    )
                    if response.status_code == 200:
                        st.success("Classification successful!")
                        results = response.json()["results"]

                        # Extract max label and probability
                        max_label = max(results, key=results.get)
                        max_prob = results[max_label]

                        st.write(f"해당 인물은 **{max_label}**와 {max_prob:.2f}%로 가장 닮았습니다.")
                    else:
                        st.error(f"에러: {response.json()['error']}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")

# Tab 2: Image Comparison
with tab2:
    st.header("닮은 꼴 찾기")

    # Display sample images
    st.subheader("나와 비교해보기")
    col1, col2, col3 = st.columns(3)

    SAMPLE_IMAGE_1_PATH = "./assets/cha.jpeg"
    SAMPLE_IMAGE_2_PATH = "./assets/hanni.jpeg"
    SAMPLE_IMAGE_3_PATH = "./assets/cat.jpg"

    with col1:
        st.image(SAMPLE_IMAGE_1_PATH, caption="Sample 1: 남자 연예인", use_container_width=True)

    with col2:
        st.image(SAMPLE_IMAGE_2_PATH, caption="Sample 2: 여자 연예인", use_container_width=True)

    with col3:
        st.image(SAMPLE_IMAGE_3_PATH, caption="Sample 3: 고양이", use_container_width=True)

    # Sample selection
    sample_choice = st.radio(
        "샘플을 선택하세요 (선택 시 1번째 사진으로 사용됩니다):",
        options=["None", "Sample 1: 남자 연예인", "Sample 2: 여자 연예인", "Sample 3: 고양이"],
        horizontal=True,
    )

    # Map sample choice to image path
    sample_image_map = {
        "Sample 1: 남자 연예인": SAMPLE_IMAGE_1_PATH,
        "Sample 2: 여자 연예인": SAMPLE_IMAGE_2_PATH,
        "Sample 3: 고양이": SAMPLE_IMAGE_3_PATH,
    }
    selected_sample = sample_image_map.get(sample_choice, None)

    # Upload images
    uploaded_image1 = st.file_uploader("첫번째 사진을 올려주세요.", type=["jpg", "jpeg", "png"], key="file1")
    uploaded_image2 = st.file_uploader("두번째 사진을 올려주세요.", type=["jpg", "jpeg", "png"], key="file2")

    # Use selected sample image if chosen, else uploaded image
    image1 = selected_sample if selected_sample else uploaded_image1
    image2 = uploaded_image2

    if st.button("사진 비교하기"):
        if not image1:
            st.error("1번째 사진을 선택하거나 업로드해야 합니다.")
        elif not image2:
            st.error("2번째 사진을 선택하거나 업로드해야 합니다.")
        else:
            # Perform image comparison
            with st.spinner("이미지 비교 중..."):
                try:
                    files = {
                        "file1": (os.path.basename(image1), open(image1, "rb")) if isinstance(image1, str) else ("uploaded1", image1),
                        "file2": (os.path.basename(image2), open(image2, "rb")) if isinstance(image2, str) else ("uploaded2", image2)
                    }
                    response = requests.post(
                        f"{SERVER_URL}/compare-images",
                        files=files
                    )
                    if response.status_code == 200:
                        st.success("비교 완료!")
                        similarity_score = response.json()["similarity_score"]
                        st.write(f"첫 번째 사진과 두 번쨰 사진은 **{similarity_score:.2f}**% 만큼 닮았습니다.")
                    else:
                        st.error(f"에러: {response.json()['error']}")
                except Exception as e:
                    st.error(f"API 연결 오류: {str(e)}")

# Tab 3: Text-to-Image Generator
with tab3:
    st.header("오늘의 선물 받아보기")

    # Prompt input
    prompt = st.text_input("받고 싶은 선물에 대해 알려주세요.")
    st.markdown(
        "<p style='font-size:12px; color:gray;'>** 예시: A ceramic coffee mug, white with gold patterns, on a wooden table, natural lighting</p>",
        unsafe_allow_html=True
    )

    if st.button("Generate Image"):
        if not prompt.strip():
            st.error("프롬프트를 입력해주세요.")
        else:
            # Perform text-to-image generation
            with st.spinner("이미지 생성 중.."):
                try:
                    response = requests.post(
                        f"{SERVER_URL}/text-to-image",
                        data={"prompt": prompt}
                    )
                    if response.status_code == 200:
                        st.success("Image generated successfully!")
                        image = Image.open(io.BytesIO(response.content))
                        st.image(image, caption="Generated Image", use_container_width=True)
                    else:
                        st.error(f"Error: {response.json()['error']}")
                except Exception as e:
                    st.error(f"API 연결 오류: {str(e)}")
