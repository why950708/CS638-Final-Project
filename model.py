
# coding: utf-8

# In[34]:
from datetime import datetime 
import tensorflow as tf
from coco_utils import load_coco_data, decode_captions
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os
import sys

class Config(object):
    def __init__(self):
        self.vocab_size =1004
        self.batch_size = 32
        self.initializer_scale =0.08
        self.H = 512 #hidden dimension
        self.T = 16 # caption length
        self.feature_len = 512
        self.W = 512 # embedding size
        self.num_epochs_per_decay = 8
        self.total_instances = 400135
        self.initial_learning_rate = 2.0
        self.input_len = 16
        self.clip_gradients = 5.0
        self.num_epochs = 1

def minibatch(data, index, batch_size=32, split='train'):
    size = data['%s_captions'%split].shape[0]
    mask = np.random.choice(size, batch_size)
    begin = batch_size*index
    caption = data['%s_captions'%split][mask]
    image_idxs = data['%s_image_idxs'%split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    caption_in = caption[:,:-1]
    caption_out = caption[:,1:]
    mask = (caption_out != 0)
    return caption_in, caption_out, mask, image_features, urls

class LSTM_Model:
    def __init__(self, mode, config):
        self.config = config
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")        
    
    def _build_input(self):
        with tf.variable_scope("input"):
            self.caption_in = tf.placeholder(tf.int32,[self.config.batch_size, self.config.input_len], name="caption_in")
            self.caption_out = tf.placeholder(tf.int32,[self.config.batch_size, self.config.input_len], name="caption_out")
            self.caption_mask = tf.placeholder(tf.int32,[self.config.batch_size, self.config.input_len], name="caption_mask")
            self.image_feature = tf.placeholder(tf.float32,[self.config.batch_size, self.config.feature_len], name="image_feature")
    
    def _build_embedding(self):
        with tf.variable_scope("word_embedding"):
            self.embed_map = tf.get_variable(name="embed_map", 
                                           shape=[self.config.vocab_size, self.config.W],
                                           initializer = self.initializer)
            word_vectors = tf.nn.embedding_lookup(self.embed_map, self.caption_in)
            self.word_embedding = word_vectors
        
        with tf.variable_scope("image_embedding"):
            feature_embedding = tf.contrib.layers.fully_connected(
            inputs=self.image_feature,
            num_outputs= self.config.H,
            activation_fn=None,
            weights_initializer= self.initializer,
            biases_initializer=None)
        
        self.feature_embedding = feature_embedding
    
    def _build_model(self):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units = self.config.H, state_is_tuple =True)
            
            # drop out is not included
            with tf.variable_scope("lstm", initializer = self.initializer) as lstm_scope:
                zero_state = lstm_cell.zero_state(
                    batch_size=self.config.batch_size,dtype=tf.float32)
                _, initial_state = lstm_cell(self.feature_embedding, zero_state)
                
                lstm_scope.reuse_variables()
                sequence_len = tf.reduce_sum(self.caption_mask,1)
                lstm_out, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                  inputs = self.word_embedding,
                                                  sequence_length = sequence_len,
                                                  initial_state = initial_state,
                                                  dtype = tf.float32,
                                                  scope = lstm_scope)
            lstm_out = tf.reshape(lstm_out, [-1, lstm_cell.output_size])
            
            with tf.variable_scope("loss"):
                w = tf.get_variable('w', [lstm_cell.output_size, self.config.vocab_size], initializer=self.initializer)
                b = tf.get_variable('b', [self.config.vocab_size], initializer=tf.constant_initializer(0.0))
                # (Nt)*H ,H*v =Nt.V, bias is zero
                logits = tf.matmul(lstm_out,w)+b
            
            
                targets = tf.reshape(self.caption_out,[-1])
                mask = tf.to_float(tf.reshape(self.caption_mask,[-1]))
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                         logits = logits)
                batch_loss = tf.div(tf.reduce_sum(tf.multiply(loss, mask)),
                                    tf.reduce_sum(mask),
                                    name="batch_loss")
                tf.losses.add_loss(batch_loss)
                self.total_loss = tf.losses.get_total_loss()
                
            with tf.name_scope("summaries"):
                #tf.summary.scalar("loss", self.batch_loss)
                tf.summary.scalar("loss", self.total_loss)
                tf.summary.histogram("histogram loss", self.total_loss)
                self.summary_op = tf.summary.merge_all()
                
    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            # decay learning rate
            num_batches = self.config.total_instances/self.config.batch_size
            decay_steps = int(num_batches*self.config.num_epochs_per_decay)
            learning_rate = self.config.initial_learning_rate


            def _decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(learning_rate,
                                                 global_step,
                                                 decay_steps = decay_steps,
                                                 decay_rate=0.5,
                                                 staircase=True)

            learning_rate_decay_fn = _decay_fn
            self.optimizer = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                      global_step = self.global_step,
                                                      learning_rate = learning_rate,
                                                      optimizer = 'SGD',
                                                      clip_gradients = self.config.clip_gradients,
                                                      learning_rate_decay_fn =learning_rate_decay_fn)



    def build_graph(self):
        #tf.reset_default_graph()
        self._build_input()
        self._build_embedding()
        self._build_model()
        self._create_optimizer()

def train_model(model, config, data):
    
    #g = tf.Graph()
    #with g.as_default():
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # if checkpoint exist, restore
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 100 epoch
        total_runs = int((config.total_instances/config.batch_size)*config.num_epochs)
        initial_step = model.global_step.eval()
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        time_now = datetime.now()

        for t in range(total_runs):

            caption_in, caption_out, mask, image_features, urls = minibatch(data,t)
            # feed data
            feed_dict = {model.image_feature: image_features, model.caption_in: caption_in, 
                        model.caption_out: caption_out, model.caption_mask: mask}
            total_loss,_,summary = sess.run([model.total_loss, model.optimizer, model.summary_op],
                                           feed_dict = feed_dict)

            writer.add_summary(summary, global_step=t)

            # print loss infor
            if(t+1) % 20 == 0:
                print('(Iteration %d / %d) loss: %f, and time eclipsed: %.2f minutes' % (
                    t + 1, total_runs, float(total_loss), (datetime.now() - time_now).seconds/60.0))
            
            #print image
            

            #save model
            if(t+1)%50 == 0 or t == (total_runs-1):
                if not os.path.exists('checkpoints/lstm'):
                    os.makedirs('checkpoints/lstm')
                saver.save(sess, 'checkpoints/lstm', t)
        
        # visualize embed matrix
        #code to visualize the embeddings. uncomment the below to visualize embeddings
        final_embed_matrix = sess.run(model.embed_map)
        
        # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        
        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = 'processed/vocab_1000.tsv'

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)


def main():
    config = Config()
    data = load_coco_data()
    model = LSTM_Model('train', config)
    model.build_graph()
    train_model(model, config, data)

main()
            
                
                
            
            
            

                
        
            
        

