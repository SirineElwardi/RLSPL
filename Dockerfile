# Stage 1: Base Image a
FROM python:3.10-slim AS base

RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /ReinforcementLearningSPL
RUN apt update && apt upgrade -y && apt install -y --no-install-recommends \
    build-essential \
    libssl-dev && \
    apt-get -f install -y --no-install-recommends && \
    apt-get clean && \

# Stage 2: Install Python Dependencies
FROM base AS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Stage 3: Final Application Stage
FROM dependencies AS final
COPY . .
EXPOSE 80
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production
# Default command
CMD ["python3", "main.py"]
