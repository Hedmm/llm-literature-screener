# CPU image with the lls command line, the web app and the transformers backend.
#
# CPU inference suits small models and the mock backend; for a 7B model, point the openai
# backend at a GPU server (vLLM, llama.cpp, Ollama) or run on a GPU machine instead.
#
#   docker build -t llm-literature-screener .
#   docker run --rm -v "$PWD:/work" llm-literature-screener screen \
#       --records /app/examples/records.ris --criteria /app/examples/protocol.yaml \
#       --labels /app/examples/labels.csv --backend mock --out results/
#   docker run --rm -p 8501:8501 llm-literature-screener ui --headless

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

# The CPU build of PyTorch is far smaller than the default CUDA build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" torch

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
RUN pip install ".[hf,ui]"

RUN useradd --create-home --uid 1000 screener \
    && mkdir -p /models /work \
    && chown screener /models /work
USER screener
WORKDIR /work

EXPOSE 8501
ENTRYPOINT ["lls"]
CMD ["--help"]
