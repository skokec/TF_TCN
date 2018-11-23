# Dependency: https://github.com/skokec/DAU-ConvNet.git (branch single-dimension)

# using smaller number of hidden units (256) with additional conv1x1 that combines dau_conv1d output and input
# this results in similar performance but is faster for development purpuse

# PTB: 1.317 after 300 epoch
python char_cnn_test.py --gpu=0 --save_path=./output/tDAU/half_gauss_init_std=2_sigma0.5_mulr=100_lr=10 --use_dau --use_conv1x1 --nhid=256 --emsize=128 --epoch=300 --lr=10

# PTB: 1.321 after 300 epoch (using sigma=0.3 and mu_learning_rate_factor=500 that need to be manually changed in the code)
python char_cnn_test.py --gpu=0 --save_path=./output/tDAU/half_gauss_init_std=2 --use_dau --use_conv1x1 --nhid=256 --emsize=128 --epoch=300 --lr=4


# using number of hidden units close to reported paper (416) - no conv1x1

# PTB: 1.322 after 300 epoch (using sigma=0.3 and mu_learning_rate_factor=500 that need to be manually changed in the code)
python char_cnn_test.py --gpu=0 --save_path=./output/tDAU/half_gauss_feat=416_init_std=2_no_conv1x1 --use_dau --nhid=416 --emsize=128 --epoch=300 --lr=4