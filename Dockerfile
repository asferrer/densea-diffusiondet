FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/densea-diffusiondet

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# detectron2 at the commit used for the results (compiled with CUDA)
ARG DETECTRON2_COMMIT=8a9d885b3d4dcf1bef015f0593b872ed8d32b4ab
RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"

COPY . .

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["sleep", "infinity"]
