FROM dolfinx/dolfinx:v0.7.3

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.2.0+cpu

RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    --index-url ${TORCH_INDEX}

RUN pip install --no-cache-dir \
    "scipy>=1.11" \
    "matplotlib>=3.8"

WORKDIR /workspace

# Examples:
#   python3 run_ssn.py --mesh 64 --output ssn_result.npz
#   python3 run_ssn.py --dim 3 --mesh 24 --solver amg
#   python3 run_xpinn.py --input ssn_result.npz
#   python3 run_pipeline.py
