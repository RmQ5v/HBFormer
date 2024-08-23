# HBFormer
>
> [**HBFormer: a single-stream framework based on hybrid attention mechanism for identification of human-virus protein-protein interactions**]

### Installation

Setup conda environment:
```bash
# Create environment
conda create -n HBFormer python=3.8 -y
conda activate HBFormer

# Instaill requirements
conda install pytorch==1.8.1 torchvision==0.9.1 -c pytorch -y

# Clone HBFormer
git clone https://github.com/RmQ5v/HBFormer.git
cd HBFormer

# Install other requirements
pip install bio-embeddings[all]
pip install -r requirements.txt
```

### Data Preparation

Prepare the benchmark(http://kurata35.bio.kyutech.ac.jp/LSTM-PHV/download_page) dataset in `<BASE_FOLDER>/data_full/`.
Download the annotation file() in `<BASE_FOLDER>/data_full/`.

### Testing

download checkpoints() to `<BASE_FOLDER>/checkpoints/`

```bash
python -m torch.distributed.launch --nproc_per_node=4 eval.py \
    --output_dir /path/to/output --batch_size 256 --embed_method pt5 \
    --folder_num $folder_number --width 360 --ckpt /path/to/checkpoint \
    --add_fea /path/to/annotation
```

### Training

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir /path/to/output --batch_size 256 --embed_method pt5 --depth 1 \
    --folder_num $folder_number --width 360 --epochs 20 --lr 8e-4 --warmup-lr 5e-4 --min-lr 1e-4 --num_heads 12 \
    --add_fea /path/to/annotation
```
