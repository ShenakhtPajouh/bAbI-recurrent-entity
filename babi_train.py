# -*- coding: utf-8 -*-
"""bAbI_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jf1iL435RbLHdUxkZlZ0A78OWjDxVUEt
"""

from google.colab import drive
drive.mount('/content/drive')

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

import tensorflow as tf
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow.contrib import autograph
from tensorflow.contrib import layers
from functools import partial

K = tf.keras.backend


def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    input_tensor: float Tensor to perform activation.
    Returns:
    `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


class Sent_encoder(tf.keras.layers.Layer):
    def __init__(self, name=None):
        if name is None:
            name = 'sent_encoder'
        super().__init__(name=name)
        self.positional_mask = None
        self.built = False

    def build(self, input_shape):
        input_shapee = input_shape.as_list()
        print(input_shapee)
        print(type(input_shapee[1]))
        self.positional_mask = self.add_weight(shape=[input_shapee[1], input_shapee[2]], name='positional_mask',
                                               dtype=tf.float32, trainable=True,
                                               initializer=tf.keras.initializers.TruncatedNormal())
        self.built = True

    def call(self, inputs, mode):
        """
        Description:
            encode given sentences with weigthed bag of words algorithm
        Args:
            input: sents shape: [current_prgrphs_num,max_sent_len,embedding_dim]
            output: encoded sentences of shape [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        """
        ' I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '
        print("positional", self.positional_mask.shape)
        if mode == 0:
            to_return = tf.reduce_sum(tf.multiply(inputs, self.positional_mask), axis=1)
        else:
            to_return = tf.reduce_sum(tf.multiply(inputs, self.positional_mask), axis=2)
        return to_return


class EntityCell(tf.keras.layers.Layer):
    """
    Entity Cell.
    call with inputs and keys
    """

    def __init__(self, max_entity_num, entity_embedding_dim, activation=tf.sigmoid, name=None,
                 **kwargs):
        if name is None:
            name = 'Entity_cell'
        super().__init__(name=name)
        self.max_entity_num = max_entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation

        self.U = None
        self.V = None
        self.W = None
        self.built = False

    def build(self, input_shape):
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = self.add_weight(shape=shape, name='U', dtype=tf.float32, trainable=True,
                                 initializer=tf.keras.initializers.TruncatedNormal())
        self.V = self.add_weight(shape=shape, name='V', dtype=tf.float32, trainable=True,
                                 initializer=tf.keras.initializers.TruncatedNormal())
        self.W = self.add_weight(shape=shape, name='W', dtype=tf.float32, trainable=True,
                                 initializer=tf.keras.initializers.TruncatedNormal())
        self.built = True

    def get_gate(self, encoded_sents, current_hiddens, current_keys):
        """
        Description:
            calculate the gate g_i for all hiddens of given paragraphs
        Args:
            inputs: encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]

            output: gates of shape : [curr_prgrphs_num, entity_num]
        """

        
        return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.expand_dims(encoded_sents, 1), current_hiddens) +
                                        tf.multiply(tf.expand_dims(encoded_sents, 1), current_keys), axis=2))

    def update_hidden(self, gates, current_hiddens, current_keys, encoded_sents):
        """
        Description:
            updates hidden_index for all prgrphs
        Args:
            inputs: gates shape: [current_prgrphs_num, entity_num]
                    encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]
        """
        curr_prgrphs_num = tf.shape(current_hiddens)[0]
        h_tilda = self.activation(
            tf.reshape(tf.matmul(tf.reshape(current_hiddens, [-1, self.entity_embedding_dim]), self.U) +
                       tf.matmul(tf.reshape(current_keys, [-1, self.entity_embedding_dim]), self.V) +
                       tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents, 1), [1, self.max_entity_num, 1]),
                                            shape=[-1, self.entity_embedding_dim]), self.W),
                       shape=[curr_prgrphs_num, self.max_entity_num, self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        updated_hiddens = current_hiddens + tf.multiply(
            tf.tile(tf.expand_dims(gates, axis=2), [1, 1, self.entity_embedding_dim]), h_tilda)

        return self.normalize(updated_hiddens)

    def normalize(self, hiddens):
        return tf.nn.l2_normalize(hiddens, axis=2)

    def call(self, inputs, prev_states, keys, use_shared_keys=False, **kwargs):
        """

        Args:
            inputs: encoded_sents of shape [batch_size, encoding_dim] , batch_size is equal to current paragraphs num
            prev_states: tensor of shape [batch_size, key_num, dim]
            keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
            use_shared_keys: if it is True, it use shared keys for all samples.

        Returns:
            next_state: tensor of shape [batch_size, key_num, dim]
        """

        encoded_sents = inputs
        gates = self.get_gate(encoded_sents, prev_states, keys)
        updated_hiddens = self.update_hidden(gates, prev_states, keys, encoded_sents)
        return updated_hiddens


    def get_initial_state(self):
        return tf.zeros([self.max_entity_num, self.entity_embedding_dim], dtype=tf.float32)


def simple_entity_network(inputs, keys, entity_cell=None,
                          initial_entity_hidden_state=None,
                          use_shared_keys=False, return_last=True):
    """
    Args:
        entity_cell: the EntityCell
        inputs: a list containing a tensor of shape [batch_size, seq_length, dim] and its mask of shape [batch_size, seq_length]
                batch_size=current paragraphs num, seq_length=max number of senteces
        keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
        use_shared_keys: if it is True, it use shared keys for all samples.
        mask_inputs: tensor of shape [batch_size, seq_length] and type tf.bool
        initial_entity_hidden_state: a tensor of shape [batch_size, key_num, dim]
        return_last: if it is True, it returns the last state, else returns all states

    Returns:
        if return_last = True then a tensor of shape [batch_size, key_num, dim] (entity_hiddens)
        else of shape [batch_size, seq_length+1 , key_num, dim] it includes initial hidden states as well as states for each step ,total would be seq_len+1
    """

    encoded_sents, mask = inputs
    seq_length = tf.shape(encoded_sents)[1]
    batch_size = tf.shape(encoded_sents)[0]
    key_num = tf.shape(keys)[1]
    entity_embedding_dim = tf.shape(keys)[2]

    if entity_cell is None:
        entity_cell = EntityCell(max_entity_num=key_num, entity_embedding_dim=entity_embedding_dim,
                                 name='entity_cell')

    if initial_entity_hidden_state is None:
        initial_entity_hidden_state = tf.tile(tf.expand_dims(entity_cell.get_initial_state(), axis=0),
                                              [batch_size, 1, 1])
    if return_last:
        entity_hiddens = initial_entity_hidden_state
    else:
        all_entity_hiddens = tf.expand_dims(initial_entity_hidden_state, axis=1)

    def cond(encoded_sents, mask, keys, entity_hiddens, i, iters):
        return tf.less(i, iters)

    def body_1(encoded_sents, mask, keys, entity_hiddens, i, iters):
        indices = tf.where(mask[:, i])
        indicess = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indicess)
        curr_keys = tf.gather(keys, indicess)
        prev_states = tf.gather(entity_hiddens, indicess)
        updated_hiddens = entity_cell(curr_encoded_sents, prev_states, curr_keys)
        entity_hiddenss = entity_hiddens + tf.scatter_nd(tf.expand_dims(indicess, 1), updated_hiddens - prev_states,
                                                         tf.shape(keys))
        return [encoded_sents, mask, keys, entity_hiddenss, tf.add(i, 1), iters]

    def body_2(encoded_sents, mask, keys, all_entity_hiddens, i, iters):
        indices = tf.where(mask[:, i])
        indicess = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indicess)
        curr_keys = tf.gather(keys, indicess)
        prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indicess)
        updated_hiddens = tf.expand_dims(entity_cell(curr_encoded_sents, prev_states, curr_keys), axis=1)
        all_entity_hiddenss = tf.concat([all_entity_hiddens,
                                         tf.scatter_nd(tf.expand_dims(indicess, 1), updated_hiddens,
                                                       [batch_size, 1, key_num, entity_embedding_dim])], axis=1)
        return [encoded_sents, mask, keys, all_entity_hiddenss, tf.add(i, 1), iters]

    i = tf.constant(0)
    if return_last:
        encoded_sentss, maskk, keyss, entity_hiddenss, ii, iterss = tf.while_loop(cond, body_1,
                                                                                  [encoded_sents, mask, keys,
                                                                                   entity_hiddens, i, seq_length])
        to_return = entity_hiddenss
    else:
        encoded_sentss, maskk, keyss, all_entity_hiddenss, ii, iterss = tf.while_loop(cond, body_2,
                                                                                      [encoded_sents, mask, keys,
                                                                                       all_entity_hiddens, i,
                                                                                       seq_length]
                                                                                      , shape_invariants=[
                encoded_sents.get_shape(), mask.get_shape(), keys.get_shape(),
                tf.TensorShape(
                    [encoded_sents.shape[0], None, keys.shape[1],
                     keys.shape[2]]),
                i.get_shape(), seq_length.get_shape()])
        to_return = all_entity_hiddenss

    return to_return


class BasicRecurrentEntityEncoder(tf.keras.layers.Layer):
    def __init__(self, embedding_matrix, max_entity_num=None, entity_embedding_dim=None, entity_cell=None, name=None,
                 **kwargs):
        if name is None:
            name = 'BasicRecurrentEntityEncoder'
        super().__init__(name=name)
        if entity_cell is None:
            if entity_embedding_dim is None:
                raise AttributeError('entity_embedding_dim should be given')
            if max_entity_num is None:
                raise AttributeError('max_entity_num should be given')
            entity_cell = EntityCell(max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                     name='entity_cell')
        self.entity_cell = entity_cell
        self.embd_matrix = embedding_matrix
        print(type(entity_embedding_dim))
        self.sent_encoder_module = Sent_encoder()
        self.built = False

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(shape=[22, 100], name='embedding_matrix', dtype=tf.float32,
                                                trainable=True, initializer=tf.keras.initializers.TruncatedNormal())
        self.built = True

    def get_embeddings(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

    def call(self, inputs, keyss, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True, **kwargs):
        """
        Args:
            inputs: paragraph, paragraph mask in a list , paragraph of shape:[batch_size, max_sents_num, max_sents_len,
            keyss: entity keys of shape : [batch_size, max_entity_num, entity_embedding_dim]
            initial_entity_hidden_state
            use_shared_keys: bool
            return_last: entity_cell and encoded_sents of shape [batch_size, max_num_sent, sents_encoding_dim]
        """

        print("in call")
        if len(inputs) != 3:
            raise AttributeError('expected 3 inputs but', len(inputs), 'were given')
        prgrphh, prgrph_maskk, questionn = inputs
        prgrph = tf.convert_to_tensor(prgrphh)
        prgrph_mask = tf.convert_to_tensor(prgrph_maskk)
        question = tf.convert_to_tensor(questionn)
        batch_size = tf.shape(prgrph)[0]
        max_sent_num = tf.shape(prgrph)[1]
        prgrph_embeddings = self.get_embeddings(prgrph)

        encoded_question = self.sent_encoder_module(self.get_embeddings(question), 0)

        keys = self.get_embeddings(tf.convert_to_tensor(keyss))
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'
        print(prgrph_embeddings.shape)
        encoded_sentsss = self.sent_encoder_module(prgrph_embeddings, 1)
        
        sents_mask = prgrph_mask[:, :, 0]
        print("return_last_encoder", return_last)
        return self.entity_cell, simple_entity_network(entity_cell=self.entity_cell,
                                                       inputs=[encoded_sentsss, sents_mask],
                                                       keys=keys,
                                                       initial_entity_hidden_state=initial_entity_hidden_state,
                                                       use_shared_keys=use_shared_keys,
                                                       return_last=return_last), encoded_question



class RecurrentEntitiyDecoder(tf.keras.layers.Layer):
    def __init__(self, embedding_matrix, entity_embedding_dim, vocab_size=None, softmax_layer=None, activation=None,
                 name=None, **kwargs):
        if name is None:
            name = 'RecurrentEntitiyDecoder'
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_matrix.shape[1]
        self.entity_embedding_dim = entity_embedding_dim

        if softmax_layer is None:
            if vocab_size is None:
                raise AttributeError("softmax_layer and vocab_size can't be both None")
            self.softmax_layer = tf.keras.layers.Dense(vocab_size)

        if activation is None:
            activation = tf.sigmoid
        self.activation = activation
        self.H = None
        self.built=False

    def build(self, input_shape):
        self.H = self.add_weight(name="H", shape=[self.entity_embedding_dim, self.entity_embedding_dim],
                                 dtype=tf.float32, trainable=True, initializer=tf.keras.initializers.TruncatedNormal())
        self.built=True
        

    def attention_entities(self, query, entities, keys_mask):
        '''
        Description:
            attention on entities

        Arges:
            inputs: query shape: [curr_prgrphs_num, embedding_dim]
                    entities shape: [curr_prgrphs_num, entities_num, entitiy_embedding_dim]
                    keys_mask shape: [curr_prgrphs_num, entities_num]
            output shape: [curr_prgrphs_num, entity_embedding_dim]
        '''
        values = tf.identity(entities)
        query_shape = tf.shape(query)
        entities_shape = tf.shape(entities)
        batch_size = query_shape[0]
        seq_length = entities_shape[1]
        indices = tf.where(keys_mask)
        queries = tf.gather(query, indices[:, 0])
        keys = tf.boolean_mask(entities, keys_mask)
        attention_logits = tf.reduce_sum(tf.multiply(queries, keys), axis=-1)
        attention_logits = tf.scatter_nd(tf.where(keys_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(keys_mask, attention_logits, tf.fill([batch_size, seq_length], -20.0))
        attention_max = tf.reduce_max(attention_logits, axis=-1, keep_dims=True)
        attention_coefficients = tf.nn.softmax(attention_logits-attention_max, axis=1)
        attention = tf.multiply(tf.expand_dims(attention_coefficients, 2) , values)
        return tf.reduce_sum(attention,axis=1)
      

    def call(self, inputs, keys_mask, encoder_hidden_states=None,
             use_shared_keys=False,
             return_last=True, attention=False):
        """
        Args:
            inputs: [entity_hiddens, question] , keys_mask
            return: distribution on the guessed answer

        """

        if len(inputs) != 2:
            raise AttributeError('expected 2 inputs but', len(inputs), 'were given')

        entity_hiddens, encoded_question = inputs
        print("entity_hiddens", entity_hiddens, entity_hiddens.shape)
        entity_hiddens = tf.convert_to_tensor(entity_hiddens)
        keys_mask = tf.convert_to_tensor(keys_mask)

        u = self.attention_entities(encoded_question, entity_hiddens, keys_mask)
        print(u.shape)
        output = self.softmax_layer(self.activation(encoded_question + tf.matmul(u, self.H)))
        return output


class Model(tf.keras.Model):
    def __init__(self, embedding_matrix, max_entity_num=None, entity_embedding_dim=None,
                 entity_cell=None, vocab_size=None, softmax_layer=None, activation=None):
        super().__init__()

        self.embedding_matrix = embedding_matrix
        self.encoder = BasicRecurrentEntityEncoder(embedding_matrix=embedding_matrix,
                                                   max_entity_num=max_entity_num,
                                                   entity_embedding_dim=entity_embedding_dim,
                                                   entity_cell=entity_cell)

        self.decoder = RecurrentEntitiyDecoder(embedding_matrix=embedding_matrix,
                                               entity_embedding_dim=entity_embedding_dim,
                                               vocab_size=vocab_size,
                                               softmax_layer=softmax_layer, activation=activation)

    def call(self, inputs, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True):
        """
         inputs=[prgrph, prgrph_mask, question]
        """
        prgrph, prgrph_mask, question, keys, keys_mask = inputs
        prgrph = tf.cast(prgrph, tf.int32)
        print('prgrph', prgrph[0])
        print(type(prgrph))
        print('question', question[0])
        prgrph_mask = tf.cast(prgrph_mask, tf.bool)
        print("prgrph_mask dtype", prgrph_mask.dtype)

        question = tf.cast(question, tf.int32)
        keys_mask = tf.cast(keys_mask, tf.bool)
        keys = tf.cast(keys, tf.int32)
        print(prgrph.shape)
        print(keys_mask)
        entity_cell, entity_hiddens, encoded_question = self.encoder(inputs=[prgrph, prgrph_mask, question], keyss=keys,
                                                                     initial_entity_hidden_state=initial_entity_hidden_state,
                                                                     use_shared_keys=use_shared_keys,
                                                                     return_last=return_last)
        output = self.decoder(inputs=[entity_hiddens, encoded_question], keys_mask=keys_mask)
        print('len_variables', len(self.variables))
        return output

import numpy as np
import tensorflow as tf
import pickle
from tensorflow.contrib import seq2seq
from keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


def calculate_loss(targets, outputs):
    """
    Args:
        inputs: outputs shape : [batch_size,max_sents_num*max_sents_len, vocab_size]
                lstm_targets shape : [batch_size, max_sents_num*max_sents_len]
                mask : [batch_size, max_sents_num*max_sents_len]

    """

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,labels=targets)
    return loss


def train(prgrphs, prgrphs_mask, questions, answers, keys, keys_mask, embedding_matrix, max_entity_num,
          entity_embedding_dim, vocab_size, learning_rate, save_path, batch_size, validation_split, epochs):



      def schedule(epoch_num, lr):
        new_lr=lr
        if epoch_num!=0 and epoch_num%25==0:
          new_lr=lr/2
        return new_lr

      num_batches=((1-validation_split)*prgrphs.shape[0])//batch_size
      train_loss=[]
      train_acc=[]
      val_loss=[]
      val_acc=[]

      def save_loss(batch,logs):
        nonlocal train_loss
        nonlocal train_acc
        if batch==0:
          train_loss=[]
          train_acc=[]
        train_loss.append(logs['loss'])
        train_acc.append(logs['acc'])

      def plot(loss, acc, rangee, mode, epoch=200):
          fig=plt.figure()
          ax1=fig.add_subplot(211)
          ax1.plot(np.arange(rangee),loss)
          ax1.set_xlabel(mode+" "+str(epoch))
          ax1.set_ylabel("loss")
          ax2=fig.add_subplot(212)
          ax2.plot(np.arange(rangee),acc)
          ax2.set_ylabel("accuracy")
          plt.show()


      start_time=time.time()
      def plot_train_loss(epoch,logs):
        nonlocal start_time
        end_time=time.time()
        print("time:",end_time-start_time)
        start_time=end_time
        val_loss.append(logs['val_loss'])
        val_acc.append(logs['val_acc'])
        plot(train_loss,train_acc,num_batches,'train',epoch)
        
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      session = tf.Session(config=config)
      tf.keras.backend.set_session(session)
      
      model = Model(embedding_matrix=embedding_matrix, max_entity_num=max_entity_num,
                        entity_embedding_dim=entity_embedding_dim,
                        vocab_size=vocab_size)
      print('leraning_rate',learning_rate)
      adam = tf.keras.optimizers.Adam(lr=learning_rate)
      model.compile(optimizer=adam,
                    loss=calculate_loss,
                    metrics=['accuracy'])

      cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                       save_weights_only=True,
                                                       verbose=1)
      tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)
      

      tn_callback=tf.keras.callbacks.TerminateOnNaN()
      lr_callback=tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
      lambda_callback=tf.keras.callbacks.LambdaCallback(on_batch_end=save_loss, on_epoch_end=plot_train_loss)

      answerss=np.zeros([answers.shape[0],vocab_size],np.int32)
      answerss[np.arange(answers.shape[0]),np.squeeze(answers)]=1

      history = model.fit(x=[prgrphs, prgrphs_mask, questions, keys, keys_mask], y=answerss, batch_size=batch_size,
                          validation_split=validation_split, epochs=epochs, 
                          callbacks=[cp_callback, lambda_callback, tbCallBack,tn_callback],shuffle=True)
      
      plot(val_loss,val_acc,epochs,'validation')


      return history

with open("/content/drive/My Drive/Colab Notebooks/bAbI/q1_train_paragraphs_m.pkl","rb") as file:
  paragraphs=pickle.load(file)
with open("/content/drive/My Drive/Colab Notebooks/bAbI/q1_train_paragraphs_mask_m.pkl","rb") as file:
  paragraphs_mask=pickle.load(file)
with open("/content/drive/My Drive/Colab Notebooks/bAbI/q1_train_questions_m.pkl","rb") as file:
  questions=pickle.load(file)
with open("/content/drive/My Drive/Colab Notebooks/bAbI/q1_train_answers_m.pkl","rb") as file:
  answers=pickle.load(file)
with open('/content/drive/My Drive/Colab Notebooks/bAbI/qa1_embedding.pkl',"rb") as file:
  embedding_matrix=pickle.load(file)
  
with open('/content/drive/My Drive/Colab Notebooks/bAbI/q1_keys_m.pkl','rb') as file:
  keys=pickle.load(file)
with open('/content/drive/My Drive/Colab Notebooks/bAbI/q1_keys_mask_m.pkl','rb') as file:
  keys_mask=pickle.load(file)
with open('/content/drive/My Drive/Colab Notebooks/bAbI/config.pkl','rb') as file:
  config=pickle.load(file)
  
print("---------------")
paragraphs_num = config['paragraphs_num']
print(paragraphs_num)
max_sent_num = config['max_sent_num']
print(max_sent_num)
max_sent_len = config['max_sent_len']
print(max_sent_len)
batch_size=config['batch_size']
print(batch_size)
epochs_num=config['epochs_num']
print(epochs_num)
initial_lr=config['initial_lr']
print(initial_lr)
max_entity_num=config['max_entity_num']
print(max_entity_num)
vocab_size=config['vocab_size']
print(vocab_size)
embedding_dim=config['embedding_dim']
print(embedding_dim)
val_split=config['val_split']
print(val_split)
print("---------------")

paragraphs=paragraphs[:paragraphs_num-(paragraphs_num % batch_size)]
paragraphs_mask=paragraphs_mask[:paragraphs_num-(paragraphs_num % batch_size)]
keys=keys[:paragraphs_num-(paragraphs_num % batch_size)]
keys_mask=keys_mask[:paragraphs_num-(paragraphs_num % batch_size)]
questions=questions[:paragraphs_num-(paragraphs_num % batch_size)]
answers=answers[:paragraphs_num-(paragraphs_num % batch_size)]


answerss=np.zeros([answers.shape[0],vocab_size],np.float64)
answerss[np.arange(answers.shape[0]),np.squeeze(answers)]=1.0

answerss=(answerss*(1-(10**-5)))+(10**-5)

vocab_size=22

train(paragraphs, paragraphs_mask, questions, answers, keys, keys_mask, embedding_matrix, max_entity_num, embedding_dim, vocab_size,
              initial_lr, "/content/drive/My Drive/Colab Notebooks/bAbI", batch_size, val_split, 5)

