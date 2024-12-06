# Image Generator

OpenAI의 CLIP 모델(`clip-vit-base-patch16`)을 활용하여 Text-to-Image 또는 Image-to-Text 연관성을 기반으로 한 데모 웹 애플리케이션입니다.
FastAPI를 통해 API 서버를 구축하고, Streamlit을 활용한 간단한 웹 UI를 제공합니다.

## 주요 기능
- 텍스트를 입력하면 관련 이미지를 반환 (Text-to-Image).
- 이미지를 업로드하면 관련 텍스트를 반환 (Image-to-Text).
- 이미지 간 유사도를 비교 (Compare-Image)
- Docker를 활용한 재현 가능 환경 제공.

## 의존성 설치

1. Python 버전: 3.8 이상 (3.12 권장)
2. 필요 라이브러리 설치
   ```bash
   pip install -r requirements.txt

## `dotenv` 설정

`.env` 파일을 생성하고, 예제 파일(`.env.example`)을 참고하여 환경 변수를 정의하세요.

```
CLIP_MODEL_NAME=clip-vit-base-patch16
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
```

## 실행 방법

### 1. 로컬 실행
1. **FastAPI 서버 실행**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000

FastAPI 서버는 `http://localhost:8000`에서 확인할 수 있습니다.


2. **Streamlit 실행**
Streamlit UI는 `http://localhost:8501`에서 확인할 수 있습니다.

    
    ```bash
    streamlit run frontend/app.py --server.port 8501
    
---

### 2. Docker 실행

1. Docker 이미지를 빌드하고 실행합니다.
    
    ```bash
    docker buildx build --platform linux/amd64,linux/arm64 -t jhkim92/fastapi-streamlit-app:latest --push .
    docker run -p 8000:8000 -p 8502:8501 jhkim92/fastapi-streamlit-app:latest
    
    ```
    

---

### 3. vast.ai에서 실행

### FastAPI

vast.ai 인스턴스에서 Docker 이미지를 불러와 FastAPI 서버를 실행합니다.

```bash
docker pull jhkim92/fastapi-streamlit-app:latest
docker run -p 8000:8000 jhkim92/fastapi-streamlit-app:latest
```

### Streamlit

Streamlit은 로컬에서 실행하거나 [Streamlit Cloud](https://streamlit.io/cloud)에 배포하여 사용합니다.

---

### **Dockerfile 작성법**

vast.ai에서 사용할 수 있는 두 가지 주요 시나리오를 고려하여 작성합니다.

#### **1. 로컬에서 Docker 이미지를 빌드하여 vast.ai로 전송하는 경우**
로컬에서 Docker 이미지를 빌드한 후 vast.ai에 업로드하여 FastAPI 서버를 실행하는 예제입니다.

```Dockerfile
# Base 이미지
FROM python:3.12-slim

# Ensure necessary system updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure /tmp permissions for Vast.ai
RUN mkdir -p /tmp && chmod -R 1777 /tmp

# Set environment variables for Vast.ai compatibility
ENV DATA_DIRECTORY=/workspace

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r app/requirements.txt && pip install streamlit

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Install process manager
RUN pip install gunicorn

# Command to start both FastAPI and Streamlit
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app & streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
```

#### **2. vast.ai에서 Pytorch Docker 이미지를 기반으로 FastAPI 서버 구축**

vast.ai에서 기본 제공하는 Pytorch Docker 이미지 위에 FastAPI를 설치하고 실행하는 예제입니다.

1. vast.ai에서 `pytorch/pytorch:latest` Docker 이미지 선택.
2. 인스턴스 실행 후, SSH로 연결.
3. 컨테이너 내부에서 FastAPI 및 기타 의존성 설치:
    
    ```bash
    apt-get update
    apt-get install -y python3-pip
    pip install fastapi uvicorn transformers
    
    ```
    
4. FastAPI 서버 실행:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
