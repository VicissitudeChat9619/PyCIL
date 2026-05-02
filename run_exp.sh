#!/bin/bash
source activate base
conda deactivate
conda activate pycil

python main.py --config=./myexp/der.json &
python main.py --config=./myexp/ewc.json &
python main.py --config=./myexp/finetune.json &
python main.py --config=./myexp/icarl.json &
# python main.py --config=./myexp/il2a.json &
python main.py --config=./myexp/lwf.json &
# python main.py --config=./myexp/pass.json &

wait
