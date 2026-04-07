FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

ENV PORT=8000
EXPOSE 8000

# Start the OpenEnv server
CMD ["python", "server/app.py"]
