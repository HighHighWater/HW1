FROM python:3.10-slim

# 1. 파이썬 환경 변수 최적화
# PYTHONDONTWRITEBYTECODE: 파이썬이 .pyc 파일을 쓰지 않도록 하여 컨테이너 용량 절약
# PYTHONUNBUFFERED: 파이썬 출력을 버퍼링하지 않아 로그가 즉시 기록되도록 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 및 빌드 툴 삭제 (이미지 사이즈 최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 (캐시 최적화를 위해 코드 복사 이전에 위치)
COPY requirements.txt .

# 5. pip 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential

# 6. 보안을 위한 비특권(Non-root) 사용자 계정 생성
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# 7. 앱 코드 전체 복사 (해당 유저 권한으로)
COPY --chown=appuser:appuser ./app /app/app

# 8. 앱 실행을 위한 포트 개방
EXPOSE 8000

# 9. 앱 실행 가이드
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
