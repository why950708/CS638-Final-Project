
# coding: utf-8

# In[1]:

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# coding: utf-8
# coding: utf-8

# In[34]:
from datetime import datetime 
import tensorflow as tf
from coco_utils import load_coco_data, decode_captions
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from image_utils import image_from_url, write_text_on_image
import os
import sys

class Config(object):
    def __init__(self):
        
        self.vocab_size =1004
        self.batch_size = 10
        self.initializer_scale =0.08
        self.H = 512 #hidden dimension
        #self.T = 16 # caption length
        self.feature_len = 512
        self.W = 512 # embedding size
        self.num_epochs_per_decay = 1
        self.total_instances = 400135
        self.initial_learning_rate = 2.0
        self.input_len = 16
        self.clip_gradients = 5.0
        self.num_epochs = 10
        self.num_of_layers = 1
        
def _run_validation(sess, caption_in, features, batch_size, model, T):
    """
    Make a single gradient update for batch data. 
    """
    # Make a minibatch of training data
    
    captions_in = caption_in[:, 0].reshape(-1, 1)
    
    state = None 
    final_preds = []
    current_pred = captions_in
    mask = np.zeros((batch_size, T))
    mask[:, 0] = 1
    
    # get initial state using image feature 
    feed_dict = {model.image_feature: features}
    state = sess.run(model.initial_state, feed_dict=feed_dict)
    
    # start to generate sentences
    for t in range(T):
        feed_dict={model.caption_in: current_pred, 
                   model.initial_state : state, 
                   model.caption_mask: mask}
            
        current_pred, state = sess.run([model.preds, model.final_state], feed_dict=feed_dict)
        
        current_pred = current_pred.reshape(-1, 1)
        
        final_preds.append(current_pred)

    return final_preds

def minibatch(data, index, batch_size,total_size, split='train'):
    #batch_size = batch_size+1
    begin = batch_size*index%total_size
    end = begin+ batch_size
    if end > total_size:
        print(begin)
        end = end - total_size# minus sign
        caption_end = data['%s_captions'%split][begin:]
        caption_first = data['%s_captions'%split][:end]
        image_idxs_end = data['%s_image_idxs'%split][:end]
        image_idxs_first = data['%s_image_idxs'%split][begin:]
        image_idxs = np.append(image_idxs_end, image_idxs_first, axis=0)
        caption = np.append(caption_end, caption_first,axis=0)
    else:
        caption = data['%s_captions'%split][begin:end]
        image_idxs = data['%s_image_idxs'%split][begin:end]
    
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
        #Mode can be "train", "test", "infer"
        self.mode = mode
        #assert error if the mode is not one of the three predesigned modes.
        assert mode in ['train','test','infer']
    def is_training(self):
        return self.mode == "train"
  
    def _build_embedding(self):
        with tf.variable_scope("word_embedding"):
            self.caption_in = tf.placeholder(tf.int32,[self.config.batch_size, None], name="caption_in")
            self.embed_map = tf.get_variable(name="embed_map", 
                                           shape=[self.config.vocab_size, self.config.W],
                                           initializer = self.initializer)
            word_vectors = tf.nn.embedding_lookup(self.embed_map, self.caption_in)
        self.word_embedding = word_vectors
        
        with tf.variable_scope("image_embedding"):
            self.image_feature = tf.placeholder(tf.float32,[self.config.batch_size, self.config.feature_len], name="image_feature")
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
        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] *self.config.num_of_layers)

        # drop out is not included
        with tf.variable_scope("lstm", initializer = self.initializer) as lstm_scope:
            self.caption_out = tf.placeholder(tf.int32,[self.config.batch_size, None], name="caption_out")
            self.caption_mask = tf.placeholder(tf.int32,[self.config.batch_size, None], name="caption_mask")
            zero_state = lstm_cell.zero_state(
                batch_size=self.config.batch_size,dtype=tf.float32)
            _, self.initial_state = lstm_cell(self.feature_embedding, zero_state)

            lstm_scope.reuse_variables()
            sequence_len = tf.reduce_sum(self.caption_mask,1)
            lstm_out, self.final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                              inputs = self.word_embedding,
                                              sequence_length = sequence_len,
                                              initial_state = self.initial_state,
                                              dtype = tf.float32,
                                              scope = lstm_scope)
        #to stack batches vertically
        lstm_out = tf.reshape(lstm_out, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            #w = tf.get_variable('w', [lstm_cell.output_size, self.config.vocab_size], initializer=self.initializer)
            #b = tf.get_variable('b', [self.config.vocab_size], initializer=tf.constant_initializer(0.0))
            # (Nt)*H ,H*v =Nt.V, bias is zero
            #tf.summary.histogram("weights", w)
            #logits = tf.matmul(lstm_out,w)+b
            #variable_summaries(w)
            logits = tf.contrib.layers.fully_connected(
                inputs = lstm_out,
                num_outputs = self.config.vocab_size,
                activation_fn = None,
                weights_initializer = self.initializer,
                scope = logits_scope
            )

        with tf.variable_scope("loss"):
            #if mode == 'train':
            targets = tf.reshape(self.caption_out,[-1])
            mask = tf.to_float(tf.reshape(self.caption_mask,[-1]))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                     logits = logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(loss, mask)),
                                tf.reduce_sum(mask),
                                name="batch_loss")
            tf.losses.add_loss(batch_loss)
            self.total_loss = tf.losses.get_total_loss()
            #if mode == 'infer':
            #softmax = tf.nn.softmax(logits)
            # get the index with largest prob
            self.preds = tf.argmax(logits, 1)
            
            
            
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.total_loss)
            tf.summary.histogram("histogramforloss", self.total_loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()
            

    def build_graph(self):
        #tf.reset_default_graph()
        self._build_embedding()
        self._build_model()
        self._create_summaries()

def train_model(model, config, data):
    
    #g = tf.Graph()
    #with g.as_default():
    ################define optimizer########
    num_batches = config.total_instances/config.batch_size
    decay_steps = int(num_batches*config.num_epochs_per_decay)
    learning_rate = tf.constant(config.initial_learning_rate)

    learning_rate_decay_fn = None
    def _decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(learning_rate,
                                         global_step,
                                         decay_steps = decay_steps,
                                         decay_rate=0.5,
                                         staircase=True)

    learning_rate_decay_fn = _decay_fn
    train_op = tf.contrib.layers.optimize_loss(loss=model.total_loss,
                                              global_step = model.global_step,
                                              learning_rate = learning_rate,
                                              optimizer = 'SGD',
                                              clip_gradients = config.clip_gradients,
                                              learning_rate_decay_fn =learning_rate_decay_fn)

    ##################
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # if checkpoint exist, restore
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        
        rand_int = np.random.randint(1,100)
        caption_in, caption_out, mask, image_features, urls = minibatch(data, rand_int, config.batch_size, config.total_instances)
            
        if not os.path.exists('test_caption'):
            os.makedirs('test_caption')
        captions_pred = _run_validation(sess, caption_in, image_features, config.batch_size, model, config.input_len) # the output is size (32, 16)
        captions_pred = [unpack.reshape(-1, 1) for unpack in captions_pred]
        captions_pred = np.concatenate(captions_pred, 1)

        captions_deco = decode_captions(captions_pred, data['idx_to_word'])

        for j in range(len(captions_deco)):
            img_name = os.path.join('test_caption', 'image_{}.jpg'.format(j))
            img = image_from_url(urls[j])
            write_text_on_image(img, img_name, captions_deco[j]) 
        
        # 100 epoch
#         total_runs = int((config.total_instances/config.batch_size)*config.num_epochs)
#         initial_step = model.global_step.eval()
        
        ### initialize summary writer
#         tf.summary.scalar("learing_rate", learning_rate)
#         a = tf.summary.merge_all()
#         writer = tf.summary.FileWriter('./graphs/singlelayer_lstm', sess.graph)
         
#         time_now = datetime.now()
#         for t in range(total_runs):

#             caption_in, caption_out, mask, image_features, urls = minibatch(data,t,config.batch_size, config.total_instances)
            
#             # feed data
#             feed_dict = {model.image_feature: image_features, model.caption_in: caption_in, 
#                         model.caption_out: caption_out, model.caption_mask: mask}
#             merge_op, _, total_loss, b = sess.run([model.summary_op, train_op, model.total_loss, a],
#                                            feed_dict = feed_dict)

#             writer.add_summary(merge_op, global_step=t)
#             writer.add_summary(b, global_step=t)
            
#             # print loss infor
#             if(t+1) % 20 == 0:
#                 print('(Iteration %d / %d) loss: %f, and time eclipsed: %.2f minutes' % (
#                     t + 1, total_runs, float(total_loss), (datetime.now() - time_now).seconds/60.0))
            
#             #print image
#             if(t+1)%100 == 0:
#                 if not os.path.exists('test_caption'):
#                     os.makedirs('test_caption')
#                 captions_pred = _run_validation(sess, caption_in, image_features, 1, model, config.input_len) # the output is size (32, 16)
#                 captions_pred = [unpack.reshape(-1, 1) for unpack in captions_pred]
#                 captions_pred = np.concatenate(captions_pred, 1)

#                 captions_deco = decode_captions(captions_pred, data['idx_to_word'])

#                 for j in range(len(captions_deco)):
#                     img_name = os.path.join('test_caption', 'image_{}.jpg'.format(j))
#                     img = image_from_url(urls[j])
#                     write_text_on_image(img, img_name, captions_deco[j])              

#             #save model
#             if(t+1)%50 == 0 or t == (total_runs-1):
#                 if not os.path.exists('checkpoints/singlelayer_lstm'):
#                     os.makedirs('checkpoints/singlelayer_lstm')
#                 saver.save(sess, 'checkpoints/singlelayer_lstm', t)
        
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
#         metadata_path = './processed/matadata.tsv'
#         if not os.path.exists(metadata_path):
#             f = open(metadata_path, "w")
#             f.close()
        embedding.metadata_path = os.path.join('processed', 'metadata.tsv')

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

