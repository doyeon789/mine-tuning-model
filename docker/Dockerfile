FROM lmsysorg/sglang:latest

# 작업 디렉토리
WORKDIR /app

# 서버 실행
CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "/app/models/Qwen3-8B", \
     "--host", "0.0.0.0", \
     "--port", "30000", \
     "--dtype", "bfloat16", \
     "--mem-fraction-static", "0.85", \
     "--reasoning-parser", "qwen3"]