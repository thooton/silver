cd ~
python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install flax orbax-checkpoint optax
python3 -m pip install wandb
curl -O -L https://huggingface.co/datasets/amongglue/books3-pretok-phi-1.5-uint16/resolve/main/train_1.bin
curl -O -L https://raw.githubusercontent.com/thooton/silver/master/index.py
curl -O -L https://raw.githubusercontent.com/thooton/silver/master/silver.py
wandb login $1
nohup python3 index.py > log.txt &
