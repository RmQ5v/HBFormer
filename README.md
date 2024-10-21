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

Step1. Prepare the [benchmark dataset](http://kurata35.bio.kyutech.ac.jp/LSTM-PHV/download_page) in `<BASE_FOLDER>/data_full/`.

Step2. Download the [annotation file](https://drive.google.com/file/d/18YZe2UwAFkHRDh2sRUvRr5-QYeT3IQRK/view?usp=sharing) in `<BASE_FOLDER>/data_full/`.

### Training

Please set $folder_number before running the training process.

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir /path/to/output --batch_size 256 --embed_method pt5 --depth 1 \
    --folder_num $folder_number --width 360 --epochs 20 --lr 8e-4 --warmup-lr 5e-4 --min-lr 1e-4 --num_heads 12 \
    --add_fea /path/to/annotation
```

### Testing

Use the above trained weights or download [checkpoints](https://drive.google.com/file/d/1ihNe4vYhf8ZqIcau5KWxiVQvkALLmqVT/view?usp=sharing) to evaluate the model performance.

```bash
python -m torch.distributed.launch --nproc_per_node=4 eval.py \
    --output_dir /path/to/output --batch_size 256 --embed_method pt5 \
    --folder_num $folder_number --width 360 --ckpt /path/to/checkpoint \
    --add_fea /path/to/annotation
```

