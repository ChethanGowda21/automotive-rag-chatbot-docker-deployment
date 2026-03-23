# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy all code (keep structure)
COPY . .

# 5️⃣ Set Python path to root so 'app' is visible
ENV PYTHONPATH=/app

# 6️⃣ Expose port
EXPOSE 8501

# 7️⃣ Run Streamlit
CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]