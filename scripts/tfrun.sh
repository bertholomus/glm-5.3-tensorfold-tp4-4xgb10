#!/bin/bash
# Run a TensorFold command in a throwaway NVIDIA PyTorch container on this node.
# Expects, in the current directory: TensorFold/ (branch glm-dsa-tp4). Mounts ~/models read-only at /models.
# usage: tfrun.sh NAME CMD...   (detached, --rm; logs: docker logs NAME)
NAME=$1; shift
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --rm --name "$NAME" --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK $( [ -d /dev/infiniband ] && echo --device /dev/infiniband ) \
  -v "$PWD":/tfw -v "$HOME/models":/models:ro -v "$PWD/cache":/root/.cache \
  -e PYTHONUNBUFFERED=1 -e TF_TP_WORLD=${TF_TP_WORLD:-1} -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-} -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-5} \
  -e NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-} -w /tfw \
  nvcr.io/nvidia/pytorch:26.07-py3 bash -c "pip install --no-deps --no-build-isolation -q -e /tfw/TensorFold >/dev/null 2>&1; $*"
