1. MNIST
- `python test_byz_all.py --dataset mnist --lr 0.01 --batch_size 32 --nworkers 100 --nbyz 20 --niter 2000 --dataset mnist --max_workers 10`
- `python test_byz_all.py --dataset mnist --lr 0.01 --batch_size 32 --nworkers 100 --nbyz 20 --niter 2000 --dataset mnist --runner foggytrust --max_workers 10`


2. Fashion-MNIST
- `python test_byz_all.py --dataset FashionMNIST --lr 0.192 --batch_size 32 --nworkers 100 --nbyz 20 --niter 2500 --max_workers 5`
- `python test_byz_all.py --dataset FashionMNIST --lr 0.192 --batch_size 32 --nworkers 100 --nbyz 20 --niter 2500 --runner foggytrust --max_workers 3`

3. Snapshot Safari (KAR, KRU, SER)
- `python test_byz_all.py --dataset SnapshotSafari --lr 0.01 --batch_size 64 --nworkers 30 --nbyz 6 --niter 300 --runner all --fog_num_groups 3 --snapshot_metadata_path ../data/snapshot/snapshot_safari_2024_metadata.json --snapshot_images_root ../data/snapshot/images --snapshot_subset_projects KAR,KRU,SER --snapshot_min_category_frequency 20 --snapshot_max_train_samples 12000 --snapshot_max_test_samples 3000 --snapshot_split_seed 7 --max_workers 10`

4. CIFAR-10
- `python test_byz_all.py --dataset CIFAR-10 --lr 0.0128 --batch_size 64 --nworkers 100 --nbyz 20 --niter 1500 --runner foggytrust --max_workers 3`

