FROM python:3.10

WORKDIR /icfc

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies (--timeout 1000 and --retries 10 prevents read timeouts on large packages like torch)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 4000 --retries 10 -r requirements.txt && \
    python -m spacy download nl_core_news_lg

# Pre-download HuggingFace models so they are baked into the Docker image
# We initialize the models on CPU just to force the download from HF Hub
RUN python -c "from simpletransformers.classification import MultiLabelClassificationModel, ClassificationModel; \
MultiLabelClassificationModel('roberta', 'CLTL/icf17-domains', use_cuda=False); \
ClassificationModel('roberta', 'CLTL/icf17-levels-domain-token', use_cuda=False); \
ClassificationModel('roberta', 'CLTL/icf17-levels', use_cuda=False)"

# Copy the rest of the application
COPY . .

# Update entrypoint to the new batching script
ENTRYPOINT ["python", "./main.py"]
