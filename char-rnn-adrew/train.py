from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os
from six.moves import cPickle

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data direcotry containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hiden state')
    parser.add_argument('--num_layers',type=int, default=2,
                        help='number of layers in RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')

    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                               'config.pkl'        : configuration;
                               'chars_vocab.pkl'   : vocabulary definitions;
                               'checkpoint'        : paths to model file(s) (created by tf).
                                                     Note: this file contains absolute paths, be careful when moving files around;
                               'model.ckpt-*'      : file(s) with model definition (created by tf)
                           """)
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size


    if args.init_from is not None:
        assert os.path.isdir(args.init_from),"%s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),\
            "chars_vocab.pkl file does not exit in path %s" %args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ['model', "rnn_size", "num_layers", "seq_length"]
        need_be_same = []