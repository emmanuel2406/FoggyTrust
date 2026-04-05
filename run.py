import os

# os.system("python test_byz_p.py --dataset mnist --lr 0.01 --batch_size 32 --nworkers 100 --nbyz 20 --byz_type trim_attack")

os.system("python test_byz_p.py --dataset mnist --lr 0.01 --batch_size 32 --nworkers 100 --nbyz 20 --byz_type trim_attack --gpu -1")

