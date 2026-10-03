#!/bin/bash
# Start full GLM-5.3 EXL3 3.0bpw on TensorFold TP4 across four GB10 nodes (throwaway containers, no restart policy).
# Rank 0 serves HTTP on 127.0.0.1:18090 (OpenAI-compatible, no auth).
# usage: tp4_start.sh [CONTEXT=32768] [EXTRA_FLAGS="--no-drafts"]     e.g. tp4_start.sh 258048 "--mtp-drafts 3"
#
# Edit for your cluster:
NODES=(spark1 spark2 spark3 spark4)          # ssh names, rank 0 first
MASTER=10.0.0.1                              # rank 0 address on the RoCE fabric
IFNAME=enp1s0f0np0                           # NCCL socket interface
# active RoCE HCAs per node (both 200G ports; names differ between GB10 units, check `ibv_devinfo`)
HCA=(mlx5_0,mlx5_2 mlx5_0,mlx5_2 rocep1s0f0,roceP2p1s0f0 rocep1s0f0,roceP2p1s0f0)
WORKDIR='~/glm53-tf'                         # holds TensorFold/ (branch glm-dsa-tp4) and tfrun.sh on every node
M=/models/GLM-5.3-EXL3-3.0bpw                # checkpoint path inside the container (tfrun.sh mounts ~/models)

CTX=${1:-32768}
# the engine's TF_GLM_* settings in this shell (e.g. TF_GLM_KV=q5) go to every rank's container
GLMENV=$(env | grep -E '^TF_GLM_[A-Z0-9_]+=[A-Za-z0-9_.,:+-]*$' | tr '\n' ' ')
EXTRA=${2:---no-drafts}
# Stop every rank's old lane first and let the nodes take the memory back: each rank's startup admission sizes the
# cache from free memory, and a just-stopped lane still holding some of it costs ~10K tokens of window.
for n in 0 1 2 3; do ssh -o BatchMode=yes -n "${NODES[$n]}" 'docker rm -f tf-tp4 >/dev/null 2>&1'; done
sleep 20
for r in 3 2 1 0; do
  if [ $r -eq 0 ]; then
    cmd="tensorfold serve $M --tp 4 --rank 0 --master $MASTER --master-port 29661 $EXTRA --context $CTX --host 127.0.0.1 --port 18090 --name GLM-5.3-EXL3-3.0bpw --temperature 0"
  else
    cmd="tensorfold serve $M --tp 4 --rank $r --master $MASTER --master-port 29661 $EXTRA --context $CTX"
  fi
  ssh -o BatchMode=yes -n "${NODES[$r]}" "cd $WORKDIR && NCCL_SOCKET_IFNAME=$IFNAME NCCL_IB_HCA=${HCA[$r]} NCCL_NET_PLUGIN=spcx $GLMENV TF_TP_WORLD=4 ./tfrun.sh tf-tp4 \"$cmd\" >/dev/null && echo \$(hostname) rank $r started"
done
