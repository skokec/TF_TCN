# Dependency: https://github.com/skokec/DAU-ConvNet.git (branch single-dimension)

# using smaller number of hidden units (256) with additional conv1x1 that combines dau_conv1d output and input
# this results in similar performance but is faster for development purpuse
python char_cnn_test.py --gpu=0 --save_path=./output/tDAU/half_gauss_init_std=2_sigma0.5_mulr=100_lr=10 --use_dau --nhid=256 --emsize=128 --epoch=300 --lr=10

# using number of hidden units close to reported paper (416) - no conv1x1
python char_cnn_test.py --gpu=0 --save_path=./output/tDAU/half_gauss_init_std=2_sigma0.5_mulr=100_lr=10 --use_dau --nhid=416 --emsize=128 --epoch=300 --lr=10