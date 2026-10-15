FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/train_all.py
EXPOSE 8501 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/Home.py --server.address 0.0.0.0 --server.port 8501"]
