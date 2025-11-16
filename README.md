https://pytorch.org/get-started/locally/
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

https://docs.rapids.ai/install/
uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu13==25.10.*" "dask-cudf-cu13==25.10.*" "cuml-cu13==25.10.*" \
    "cugraph-cu13==25.10.*" "nx-cugraph-cu13==25.10.*" "cuxfilter-cu13==25.10.*" \
    "cucim-cu13==25.10.*" "pylibraft-cu13==25.10.*" "raft-dask-cu13==25.10.*" \
    "cuvs-cu13==25.10.*" "nx-cugraph-cu13==25.10.*"

uv add -r requirements.txt