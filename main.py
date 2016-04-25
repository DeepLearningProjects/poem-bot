# -- coding: utf-8 --
# from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model


save_dir = 'save'
num_sample = 500
prime =  ' '

with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
    saved_args = cPickle.load(f)
with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
    chars, vocab = cPickle.load(f)
model = Model(saved_args, True)

s = tf.Session()

s.run(tf.initialize_all_variables())
saver = tf.train.Saver(tf.all_variables())
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(s, ckpt.model_checkpoint_path)
print (model.sample(s, chars, vocab, num_sample, prime))

def sample(num, prime):
    return model.sample(s, chars, vocab, num, prime)

#print (sample(200, "the"))

# webapp
from flask import Flask, jsonify, render_template, request
from flask.ext.cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/char-rnn', methods=['GET'])
def char_rnn():
    num = request.args.get('num', '200')
    prime = request.args.get('prime', ' ')
    app.logger.info(num)
    app.logger.info(prime)

    poem = sample(500, " ")
    return poem

@app.route('/')
def main():
    return "I'm here!"

if __name__ == '__main__':
   # app.debug = True
    app.run()