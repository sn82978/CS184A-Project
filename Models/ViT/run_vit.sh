#!/bin/bash
PROJECT=/shareddata/shreya/vit
export DATA_ROOT=$PROJECT # folder containing DATA_PP/ and the images
export OUT_DIR=$PROJECT/vit_results # results land here
export HF_HOME=$PROJECT/hf_cache # pretrained ViT weights are cached here
export NUM_WORKERS=8 # dataloader processes

# ---------------------------------------------------------
# before running do this:
# pip install torch torchvision transformers pandas numpy pillow scikit-learn matplotlib tqdm
# to run do this:
# nohup bash run_vit.sh > vit_nohup.log 2>&1 &
# tail -f vit_results/train.log
# after running, do this:
# rsync -avP shreya@tagide.fc.ul.pt:/shareddata/shreya/vit/vit_results/ ~/Downloads/vit_results/ <-- from mac
# ---------------------------------------------------------

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
fi
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

source $PROJECT/venv/bin/activate
mkdir -p "$OUT_DIR"

nvidia-smi || echo "No GPU visible"

cd "$PROJECT"
python -u train_vit.py 2>&1 | tee "$OUT_DIR/train.log"