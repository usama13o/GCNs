cd projects/GCN/GraphGym/run/
# python /home/uz1/projects/GCN/gen_graphs.py -size 256 --patch_size 32 --k 8 --dataset pathmnist --batch_size 30
python /home/uz1/projects/GCN/gen_graphs.py -size 256 --patch_size 32 --k 16 --dataset pathmnist --batch_size 30
python /home/uz1/projects/GCN/gen_graphs.py -size 256 --patch_size 32 --k 32 --dataset pathmnist --batch_size  30
python /home/uz1/projects/GCN/gen_graphs.py -size 256 --patch_size 32 --k 64 --dataset pathmnist --batch_size 30
python /home/uz1/projects/GCN/gen_graphs.py -size 256 --patch_size 32 --k 128 --dataset pathmnist   --batch_size  30