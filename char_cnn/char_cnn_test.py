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
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to the embedded layer (0 = no dropout) (default: 0.1)')
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--emsize', type=int, default=100,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
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
parser.add_argument('--use_conv1x1', action='store_true',
                    help='combine conv1d output with input using conv1x1 (default: false)')

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
dropout = args.dropout
emb_dropout = args.emb_dropout


if args.use_dau:
    output = TCNv1_DAU(
        input_layer,
        n_characters,
        num_chans,
        args.emsize,
        kernel_size=k_size,
        dropout=dropout_switch,
        bn_switch=bn_switch,
        use_conv1x1=args.use_conv1x1)
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

cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')

#loss_for_minimization = cross_entropy_mean

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
losses_regularization = tf.add_n(reg_losses, name='losses_regularization') if len(reg_losses) > 0 else tf.to_float(0)

loss_for_minimization = tf.add_n([cross_entropy_mean,losses_regularization], name='total_loss')

lr = tf.placeholder(dtype=tf.float32)
if args.use_dau and False:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

gradients, variables = zip(*optimizer.compute_gradients(loss_for_minimization))
if args.clip > 0:
    gradients, _ = tf.clip_by_global_norm(gradients, args.clip)
update_step = optimizer.apply_gradients(zip(gradients, variables))

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

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


def train(epoch, sess, lr_val):
    total_loss = 0
    start_time = time.time()
    losses = []
    source = train_data
    source_len = source.shape[1]
    for batch_idx, i in enumerate(range(0, source_len - 1, args.validseqlen)):

        if i + args.seq_len - args.validseqlen >= source_len:
            continue

        inp, target = get_batch(source, i, args)
        # print('INPUT SHAPE: ', inp.shape)
        # padding or discard?
        if target.shape[1] < args.seq_len:
            continue

        if i == 1 and False:
            vars = sess.run(tf.global_variables())
            grads = sess.run(gradients, feed_dict={input_layer: inp, labels: target, bn_switch: True, lr: lr_val, dropout_switch: float(dropout)})

            for i, v in enumerate(vars):
                if len(np.squeeze(v).shape) > 1:
                    print np.std(vars.flatten()), '\t\t', str(v.shape)

            for i, g in enumerate(grads):
                if len(np.squeeze(g).shape) > 1:
                    print np.std(g.flatten()), '\t\t', str(g.shape)


        _, l = sess.run(
            [update_step, loss], feed_dict={
                input_layer: inp,
                labels: target,
                bn_switch: True,
                dropout_switch: float(dropout),
                lr: lr_val,
            })

        total_loss += l.mean()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            losses.append(cur_loss)
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                'loss {:5.3f} | bpc {:5.3f}'.format(
                    epoch, batch_idx, int(
                        (source_len - 0.5) / args.validseqlen), lr_val,
                    elapsed * 1000 / args.log_interval, cur_loss,
                    cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

        # if batch % (200 * args.log_interval) == 0 and batch > 0:
        #     vloss = evaluate(val_data)
        #     print('-' * 89)
        #     print('| In epoch {:3d} | valid loss {:5.3f} | '
        #           'valid bpc {:8.3f}'.format(epoch, vloss, vloss / math.log(2)))
        #     model.train()


    return sum(losses) * 1.0 / len(losses)


def main():
    lr_val = args.lr
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        try:

            total_variables = np.sum([
                np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()
            ])
            print('Total variables {:5d}'.format(total_variables))

            global total_steps
            total_steps = 0

            print("Training for %d epochs..." % args.epochs)
            all_losses = []
            best_vloss = 1e7
            best_vloss_epoch = -20
            last_lr_drop_epoch = 0
            for epoch in range(1, args.epochs + 1):
                plot_mu_dist(sess,epoch-1)

                loss = train(epoch, sess, lr_val)
                vloss = evaluate(val_data, sess)

                print('-' * 89)
                print(
                    '| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.
                    format(epoch, vloss, vloss / math.log(2)))

                test_loss = evaluate(test_data, sess)
                print('=' * 89)
                print(
                    '| End of epoch {:3d} | test loss {:5.3f} | test bpc {:8.3f}'.
                    format(epoch, test_loss, test_loss / math.log(2)))
                print('=' * 89)

                if epoch > 50 and vloss > max(all_losses[-7:]) and epoch - last_lr_drop_epoch  > 10:
                    lr_val = lr_val / 10.
                    last_lr_drop_epoch = epoch
                all_losses.append(vloss)

                if vloss < best_vloss and epoch -best_vloss_epoch >= 10:
                    print("Dummy Saving...")
                    saver.save(sess, os.path.join(args.save_path, 'model.ckpt'), global_step=epoch)
                    best_vloss = vloss
                    best_vloss_epoch = epoch


        except KeyboardInterrupt:
            print('-' * 89)
            print("Dummy Saving before quit...")
            saver.save(sess, os.path.join(args.save_path, 'model-last.ckpt'), global_step=epoch)

        # Run on test data.
        test_loss = evaluate(test_data, sess)
        print('=' * 89)
        print(
            '| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
                test_loss, test_loss / math.log(2)))
        print('=' * 89)

def plot_mu_dist(sess, epoch):
    import pylab as plt

    g_vars = sess.run([v for v in tf.global_variables() if 'DAUConv/g' in v.name])
    w_vars = sess.run([v for v in tf.global_variables() if 'DAUConv/weight' in v.name])
    mu_vars = sess.run([v for v in tf.global_variables() if 'DAUConv/mu1' in v.name])

    f = plt.figure(figsize=(16,6))
    for i in range(len(mu_vars)):
        plt.subplot(2, 6, i + 1)
        plt.hist(np.abs(mu_vars[i].flatten()), bins=100)
        plt.title('mu ly%d' % i)

    for i in range(len(mu_vars)):
        plt.subplot(2, 6, 6 + i + 1)
        plt.hist(np.abs(mu_vars[i].flatten()), bins=100,
                 weights=np.abs(g_vars[i] * w_vars[i] / np.sum(w_vars[i] ** 2, axis=(0, 1, 2))).flatten())
        plt.title('mu*w ly%d' % i)

    plt.show(block=False)

    f.savefig(os.path.join(args.save_path,'mu_dist_epoch_%d.pdf' % epoch))
    f.savefig(os.path.join(args.save_path, 'mu_dist_epoch_%d.png' % epoch))

    plt.close()


# train_by_random_chunk()
# if __name__ == "__main__":
main()
