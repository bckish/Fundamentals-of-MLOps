FROM python:3.13-slim
WORKDIR /app
COPY model.py serve.py test_model.py ./
RUN python -m unittest -v && python model.py
ENV HOST=0.0.0.0 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8080
CMD ["python", "serve.py"]
