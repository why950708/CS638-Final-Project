#-*-coding:utf-8-*-
from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

def main():
    #一看到这个是不是特别的熟悉？没错和train.py里面的一个意思
    parser = argparse.ArgumentParser()
    #储存checkpoint，不太懂为什么sample的时候还有这个选项
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    #生成的字符个数
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    #指定一个开头，如果有开头标志的话这里可以是其他的，默认设置时" "
    parser.add_argument('--prime', type=text_type, default=u' ',
                       help='prime text')

    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def sample(args):
    #载入各种参数
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    #使用模型
    model = Model(saved_args, True)
    #let's roll
    with tf.Session() as sess:
        #初始化所有的变量
        tf.initialize_all_variables().run()
        #创建一个saver，后面模型重载
        saver = tf.train.Saver(tf.all_variables())
        #载入checkpoint
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #官方说明：Restores previously saved variables
            saver.restore(sess, ckpt.model_checkpoint_path)
            #来我们再回到model.py看一下这sample方法
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample))

if __name__ == '__main__':
    main()
def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        #let's go
        state = sess.run(self.cell.zero_state(1, tf.float32))

        #先把开头自己预设的prime_txt送进模型，不计输出
        #这一块程序段还是很好理解的
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            #前面说过，vocab是个字典
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        #weight = [0.1,0.2,0.3,0.4]
        #(分布函数)t = [0.1,0.3,0.6,1]
        #s = 1
        #为什么这样pick还不是太懂
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            #注意！！这里的probs是长度是1*65的，前面在训练的时候因为batch_size和seq_length都是50
            # 所以是2500*65之后用了这2500组预测结果来求loss，再BPTT，
            # 这里只是根据一个输入求一个输出，batch_size和seq_length都是1，因此是1*65
            # 所以p就是代表了由长度为65的一个数组，每一位代表着预测为该位的概率值
            p = probs[0]

            if sampling_type == 0:
                #第一种方法，直接取最大的prob的索引值
                sample = np.argmax(p)
            elif sampling_type == 2:
                #第二种方法，如果输入是空格，则wighted_pick
                #否则取最大prob的索引
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:
                #一直使用weighted_pick方法
                # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret