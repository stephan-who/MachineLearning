import codecs
import  os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir,batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")


        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading_preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
            self.create_batches()
            self.reset_batch_pointer()

        def preprocess(self, input_file, vocab_file, tensor_file):
            with codecs.open(input_file, "r", encoding=self.encoding) as f:
                data = f.read()
            counter = collections.Counter(data)
            count_pairs = sorted(counter.items(), key = lambda x: -x[1])
            self.chars, _= zip(*count_pairs)
            self.vocab_size = len(self.chars)
            self.vocab = dict(zip(self.chars, range(len(self.chars))))
            with open(vocab_file, 'wb') as f:
                cPickle.dump(self.chars, f)
            self.tensor = np.array(list(map(self.vocab.get, data)))
            np.save(tensor_file, self.tensor)


        def load_preprocessed(self, vocab_file, tensor_file):
            with open(vocab_file, 'rb') as f:
                self.chars = cPickle.load(f)
            self.vocab_size = len(self.chars)