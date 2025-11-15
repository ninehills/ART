FROM ghcr.io/ninehills/langimage:cuda126-251026

RUN git clone --recurse-submodules https://github.com/ninehills/ART.git

# Set shell to use bash and load conda
SHELL ["/bin/bash", "--login", "-c"]

RUN /opt/conda/bin/conda create -n art python==3.12

RUN --mount=type=cache,target=/root/.cache \
    cd ART && \
    /opt/conda/envs/art/bin/python -m pip install -e .[backend]

RUN --mount=type=cache,target=/root/.cache \
    cd ART && \
    /opt/conda/envs/art/bin/hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B && \
    /opt/conda/envs/art/bin/hf download intfloat/e5-base-v2 --local-dir models/e5-base-v2 && \
    /opt/conda/envs/art/bin/hf download --repo-type dataset yixuantt/MultiHopRAG --local-dir examples/rag/data/MultiHopRAG

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    /opt/conda/envs/art/bin/python -m pip install -r requirements.txt
COPY . ./rag
