FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

ENV PORT=7860
EXPOSE 7860

# Start the unified OpenEnv API + Gradio UI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
