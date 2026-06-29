# The source code for "Curriculum-Guided Semantic Prototype Shaping for Data-Free Knowledge Distillation"

CSP-DFKD is a model compression method that guides pseudo-sample generation through curriculum scheduling and semantic prototype constraints, reduces semantic drift, and improves the stability and performance of data-free knowledge distillation.

## Project Structure

CSP-DFKD/
├── datafree_kd.py                 # Main entry point for data-free knowledge 
├── train_scratch.py               # Train teacher or baseline models from scratch
├── registry.py                    # Dataset, model, and preprocessing registry
├── datafree/                      # Core code for DFKD, covering the method, model, loss function, evaluator and other key modules
├── label_embedding/               # Label semantic embedding files
├── checkpoints/pretrained/        # Directory for pretrained teacher checkpoints
└── data/                          # Dataset directory

## Quickstart

### 1. Setup Environment

The recommended environment used for this experiment is:
Python: 3.12
PyTorch: 2.8.0

```bash
pip install numpy pillow tqdm wandb kornia scipy scikit-learn pandas matplotlib seaborn termcolor xlsxwriter
```

### 2. Prepare Pretrained Models

To reproduce our results, please download pre-trained teacher models from [Dropbox-Models (266 MB)](https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0) and extract them as `checkpoints/pretrained`.
Instead, you can train a model from scratch as follows.
```bash
python train_scratch.py --model wrn40_2 --dataset cifar10 --batch-size 256 --lr 0.1 --epoch 200 --gpu 0
```

### 3. Reproduce our results

* To get similar results of our method on CIFAR datasets, run the following commands
    ```bash
    python datafree_kd.py --data_root ./data --dataset cifar10 --teacher wrn40_2 --student wrn16_2 \
      --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --lr_g 4e-3 --gpu 0 \
      --warmup 20 --epochs 320 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
      --adv 1.33 --bn 10.0 --oh 0.5 --curriculum --curriculum_warmup_epochs 40 \
      --conf 0.25 --proto 0.5 --div 0.15 --teach 0.2 \
      --proto_momentum 0.9 --proto_gate 0.70 --div_gate 0.70 \
      --curriculum_easy_ratio 0.25 --curriculum_mid_ratio 0.60 \
      --save_dir run/c10-w402-w162-cifar10 \
      --log_tag c10-w402-w162-cifar10
    ```

