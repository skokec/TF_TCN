from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing',
      Path(__file__).resolve())

import argparse
import sys
from utils import *
from model import TCN
from model_dau import TCN_DAU, TCNv1_DAU
import time
import math
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')
parser.add_argument('--emsize', type=int, default=100,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--nhid', type=int, default=450,
                    help='number of hidden units per layer (default: 450)')
parser.add_argument('--validseqlen', type=int, default=320,
                    help='valid sequence length (default: 320)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='total sequence length, including effective history (default: 400)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use (default: ptb)')
parser.add_argument('--save_path', type=str, default='./output/',
                    help='output folder for saved model')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id (default: 0)')
parser.add_argument('--use_dau', action='store_true',
                    help='enable DAU model (default: false)')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
tf.set_random_seed(args.seed)

print(args)
file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(
    args)

n_characters = len(corpus.dict)
train_data = batchify(char_tensor(corpus, file), args.batch_size)
val_data = batchify(char_tensor(corpus, valfile), args.batch_size)
test_data = batchify(char_tensor(corpus, testfile), args.batch_size)
print("Corpus size: ", n_characters)
print("train_data size:", train_data.shape)
print("val_data size", val_data.shape)
print("test_data size", test_data.shape)

bn_switch = tf.placeholder(dtype=tf.bool)
dropout_switch = tf.placeholder(dtype=tf.float32)

input_layer = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_len))
labels = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_len))
one_hot = tf.one_hot(labels, depth=n_characters, axis=-1, dtype=tf.float32)

num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize


if args.use_dau:
    output = TCNv1_DAU(
        input_layer,
        n_characters,
        num_chans,
        args.emsize,
        kernel_size=k_size,
        dropout=dropout_switch,
        bn_switch=bn_switch)
else:
    output = TCN(
        input_layer,
        n_characters,
        num_chans,
        args.emsize,
        kernel_size=k_size,
        dropout=dropout_switch,
        bn_switch=bn_switch)

eff_history = args.seq_len - args.validseqlen
final_output = tf.reshape(output[:, eff_history:, :], (-1, n_characters))
final_target = tf.reshape(one_hot[:, eff_history:, :], (-1, n_characters))

tf.stop_gradient(final_target)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=final_output, labels=final_target)

saver = tf.train.Saver(tf.global_variables())

def evaluate(source, sess):
    source_len = source.shape[1]
    losses = []
    for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target = get_batch(source, i, args)

        # padding or discard?
        if target.shape[1] < args.seq_len:
            continue

        l = sess.run(
            [loss], feed_dict={
                input_layer: inp,
                labels: target,
                bn_switch: False,
                dropout_switch: float(0.0),
            })

        losses.append(l)

    val_loss = np.mean(np.ndarray.flatten(np.array(losses)))
    return val_loss

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(args.save_path)

        saver.restore(sess, ckpt.model_checkpoint_path)

        # Run on test data.
        test_loss = evaluate(test_data, sess)
        print('=' * 89)
        print(
            '| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
                test_loss, test_loss / math.log(2)))
        print('=' * 89)




# train_by_random_chunk()
# if __name__ == "__main__":
main()
