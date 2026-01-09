# Use python slim image to keep size down
FROM python:3.11-slim

# 1. Install LaTeX (The heavy part)
# texlive-latex-base: Minimal latex
# texlive-fonts-recommended: Common fonts
# texlive-latex-extra: Common packages like geometry, enumitem
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code
COPY . .

# 4. Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
