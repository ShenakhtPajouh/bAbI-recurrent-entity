import tensorflow as tf
import numpy as np
from tensorflow.contrib import autograph

K = tf.keras.backend


class Sent_encoder(tf.keras.Model):
    def __init__(self, name=None):
        if name is None:
            name = 'sent_encoder'
        super().__init__(name=name)

    def call(self, inputs):
        """
        Description:
            encode given sentences with bag of words algorithm
        Args:
            input: sents shape: [current_prgrphs_num,max_sent_len,embedding_dim]
            output: encoded sentences of shape [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        """
        ' I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '

        return tf.reduce_sum(inputs, 1)


class EntityCell(tf.keras.layers.Layer):
    """
    Entity Cell.
    call with inputs and keys
    """

    def __init__(self, max_entity_num, entity_embedding_dim, activation=tf.nn.relu, name=None,
                 **kwargs):
        if name is None:
            name = 'Entity_cell'
        super().__init__(name=name)
        self.max_entity_num = max_entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation
        # if initializer is None:
        #     self.initializer = tf.keras.initializers.random_normal()

        self.U = None
        self.V = None
        self.W = None
        self.built = False

    def build(self, input_shape):
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = self.add_weight(shape=shape, name='U')
        self.V = self.add_weight(shape=shape, name='V')
        self.W = self.add_weight(shape=shape, name='W')
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

        print('enocded_sents dtype:', tf.shape(encoded_sents))
        print('current_hiddens dtype:', current_hiddens.dtype)
        print('enocded_sents shape:', tf.shape(encoded_sents))
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
        # tf.multiply(gates,h_tilda)
        print("gates shape:", tf.shape(gates))
        updated_hiddens = current_hiddens + tf.multiply(
            tf.tile(tf.expand_dims(gates, axis=2), [1, 1, self.entity_embedding_dim]), h_tilda)

        return updated_hiddens

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
        return self.normalize(updated_hiddens)

    def get_initial_state(self):
        return tf.zeros([self.max_entity_num, self.entity_embedding_dim], dtype=tf.float32)

    # def __call__(self, inputs, prev_state, keys, use_shared_keys=False, **kwargs):
    #     """
    #     Do not fill this one
    #     """
    #     return super().__call__(inputs=inputs, prev_state=prev_state, keys=keys,
    #                             use_shared_keys=use_shared_keys, **kwargs)


# @autograph.convert()
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
    print("type mask", type(mask))
    # print("encoded_sents shape:", encoded_sents.shape)
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
        print("return_lastttttttttt:", return_last)
        all_entity_hiddens = tf.expand_dims(initial_entity_hidden_state, axis=1)

    def cond(encoded_sents, mask, keys, entity_hiddens, i, iters):
        return tf.less(i, iters)

    def body_1(encoded_sents, mask, keys, entity_hiddens, i, iters):
        indices = tf.where(mask[:, i])
        indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indices)
        curr_keys = tf.gather(keys, indices)
        prev_states = tf.gather(entity_hiddens, indices)
        updated_hiddens = entity_cell(curr_encoded_sents, prev_states, curr_keys)
        entity_hiddens = entity_hiddens + tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens - prev_states,
                                                        tf.shape(keys))
        return [encoded_sents, mask, keys, entity_hiddens, tf.add(i, 1), iters]

    def body_2(encoded_sents, mask, keys, all_entity_hiddens, i, iters):
        indices = tf.where(mask[:, i])
        indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indices)
        curr_keys = tf.gather(keys, indices)
        prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
        updated_hiddens = tf.expand_dims(entity_cell(curr_encoded_sents, prev_states, curr_keys), axis=1)
        all_entity_hiddens = tf.concat([all_entity_hiddens,
                                        tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens,
                                                      [batch_size, 1, key_num, entity_embedding_dim])], axis=1)
        return [encoded_sents, mask, keys, all_entity_hiddens, tf.add(i, 1), iters]

    i = tf.constant(0)
    if return_last:
        encoded_sents, mask, keys, entity_hiddens, i, iters = tf.while_loop(cond, body_1,
                                                                            [encoded_sents, mask, keys,
                                                                             entity_hiddens, i, seq_length])
        to_return = entity_hiddens
    else:
        # print("seq_length.get_shape()",seq_length.get_shape())
        encoded_sents, mask, keys, all_entity_hiddens, i, iters = tf.while_loop(cond, body_2,
                                                                                [encoded_sents, mask, keys,
                                                                                 all_entity_hiddens, i, seq_length]
                                                                                , shape_invariants=[
                encoded_sents.get_shape(), mask.get_shape(), keys.get_shape(),
                tf.TensorShape(
                    [encoded_sents.shape[0], None, keys.shape[1],
                     keys.shape[2]]),
                i.get_shape(), seq_length.get_shape()])
        to_return = all_entity_hiddens

    return to_return


class BasicRecurrentEntityEncoder(tf.keras.Model):
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
        self.embedding_matrix = embedding_matrix
        self.fc1 = tf.keras.layers.Dense(units=tf.shape(self.embedding_matrix)[1] // 2)
        self.fc2 = tf.keras.layers.Dense(units=entity_embedding_dim)
        self.sent_encoder_module = Sent_encoder()

    # @property
    # def variables(self):
    #     return self.trainable_variables+self.entity_cell.variables

    def call(self, inputs, keys, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True, **kwargs):
        """
        Args:
            inputs: paragraph, paragraph mask in a list , paragraph of shape:[batch_size, max_sents_num, max_sents_len,
            keys: entity keys of shape : [batch_size, max_entity_num, entity_embedding_dim]
            initial_entity_hidden_state
            use_shared_keys: bool
            return_last: entity_cell and encoded_sents of shape [batch_size, max_num_sent, sents_encoding_dim]
        """

        if len(inputs) != 2:
            raise AttributeError('expected 2 inputs but', len(inputs), 'were given')
        prgrph, prgrph_mask = inputs
        prgrph_mask = tf.convert_to_tensor(prgrph_mask)
        batch_size = tf.shape(prgrph)[0]
        max_sent_num = tf.shape(prgrph)[1]
        prgrph_embeddings_0 = tf.nn.embedding_lookup(self.embedding_matrix, prgrph)
        prgrph_embeddings_0 = tf.convert_to_tensor(prgrph_embeddings_0)
        prgrph_embeddings_1 = self.fc1(prgrph_embeddings_0)
        prgrph_embeddings = self.fc2(prgrph_embeddings_1)
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'
        encoded_sents = tf.zeros([batch_size, 1, tf.shape(prgrph_embeddings)[3]])

        # for i in range(max_sent_num):
        #     ''' to see which sentences are available '''
        #     indices = tf.where(prgrph_mask[:, i, 0])
        #     # print('indices shape encode:, indices.shape)
        #     indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        #     current_sents = tf.gather(prgrph_embeddings[:, i, :, :], indices)
        #     # print('current_sents_call shape:', current_sents.shape)
        #     curr_encoded_sents = tf.expand_dims(self.sent_encoder_module(current_sents), axis=1)
        #     encoded_sents = tf.concat([encoded_sents, curr_encoded_sents], axis=1)

        def cond(prgrph_mask, prgrph_embeddings, encoded_sents, i, max_sent_num):
            return tf.less(i, max_sent_num)

        def body(prgrph_mask, prgrph_embeddings, encoded_sents, i, max_sent_num):
            indices = tf.where(prgrph_mask[:, i, 0])
            indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], indices)
            curr_encoded_sents = tf.expand_dims(self.sent_encoder_module(current_sents), axis=1)
            encoded_sents = tf.concat([encoded_sents, curr_encoded_sents], axis=1)
            return [prgrph_mask, prgrph_embeddings, encoded_sents, tf.add(i, 1), max_sent_num]

        i = tf.constant(0)
        prgrph_mask, prgrph_embeddings, encoded_sents, i, max_sent_num = \
            tf.while_loop(cond, body, [prgrph_mask, prgrph_embeddings, encoded_sents, i, max_sent_num],
                          shape_invariants=[prgrph_mask.get_shape(), prgrph_embeddings.get_shape(),
                                            tf.TensorShape([prgrph_embeddings.get_shape()[0], None,
                                                            prgrph_embeddings.get_shape()[3]]),
                                            i.get_shape(), max_sent_num.get_shape()])
        print("encoded_sents", encoded_sents)
        encoded_sents = encoded_sents[:, 1:, :]
        sents_mask = prgrph_mask[:, :, 0]
        print("return_last_encoder", return_last)
        return self.entity_cell, simple_entity_network(entity_cell=self.entity_cell, inputs=[encoded_sents, sents_mask],
                                                       keys=keys,
                                                       initial_entity_hidden_state=initial_entity_hidden_state,
                                                       use_shared_keys=use_shared_keys,
                                                       return_last=return_last)


class RecurrentEntitiyDecoder(tf.keras.layers.Layer):
    def __init__(self, embedding_matrix, fc_layer1, fc_layer2,vocab_size=None, softmax_layer=None, activation=None, name=None, **kwargs):
        if name is None:
            name = 'RecurrentEntitiyDecoder'
        super().__init__(name=name)
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = tf.shape(embedding_matrix)[1]

        self.fc1 = fc_layer1
        self.fc2 = fc_layer2

        if softmax_layer is None:
            if vocab_size is None:
                raise AttributeError("softmax_layer and vocab_size can't be both None")
            self.softmax_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

        if activation is None:
            activation = tf.nn.relu

        self.sent_encoder_module = Sent_encoder()

        self.H = None

    def build(self, input_shape):
        self.entity_attn_matrix = K.random_normal_variable(shape=[self.rnn_hidden_size, self.embedding_dim],
                                                           mean=0, scale=0.05, name='entity_attn_matrix')
        self.H = self.add_weight(name="H", shape=[self.embedding_dim, self.embedding_dim])

    def attention_entities(self, query, entities, keys_mask):
        '''
        Description:
            attention on entities

        Arges:
            inputs: query shape: [curr_prgrphs_num, rnn_hidden_size]
                    entities shape: [curr_prgrphs_num, entities_num, entitiy_embedding_dim]
                    keys_mask shape: [curr_prgrphs_num, entities_num]
            output shape: [curr_prgrphs_num, entity_embedding_dim]
        '''

        print("attention_entities, entities shape: ", entities.shape)

        values = tf.identity(entities)
        query_shape = tf.shape(query)
        entities_shape = tf.shape(entities)
        batch_size = query_shape[0]
        seq_length = entities_shape[1]
        indices = tf.where(keys_mask)
        queries = tf.gather(query, indices[:, 0])
        entities = tf.boolean_mask(entities, keys_mask)
        # print(queries.shape)
        # print(self.entity_attn_matrix.shape)
        attention_logits = tf.reduce_sum(tf.multiply(tf.matmul(queries, self.entity_attn_matrix), entities), axis=-1)
        # print('attention logits:',attention_logits)
        # print('tf.where(memory_mask):',tf.where(memory_mask))
        attention_logits = tf.scatter_nd(tf.where(keys_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(keys_mask, attention_logits, tf.fill([batch_size, seq_length], -float("Inf")))
        attention_coefficients = tf.nn.softmax(attention_logits, axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values

        return tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(query, self.entity_attn_matrix), axis=1), entities),
                             axis=1)

    def get_embeddings(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

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

        entity_hiddens, question = inputs
        entity_hiddens = tf.convert_to_tensor(entity_hiddens)
        question = tf.convert_to_tensor(question)
        keys_mask = tf.convert_to_tensor(keys_mask)

        embedded_qeustion_0 = self.get_embeddings(question)
        embedded_qeustion_1 = self.fc1(embedded_qeustion_0)
        embedded_qeustion = self.fc2(embedded_qeustion_1)

        u = self.attention_entities(embedded_qeustion, entity_hiddens, keys_mask)
        output = self.softmax_layer(embedded_qeustion + tf.matmul(self.H, u))
        return output


class Model(tf.keras.Model):
    def __init__(self, embedding_matrix, max_entity_num=None, entity_embedding_dim=None,
                 entity_cell=None, vocab_size=None, softmax_layer=None, activation=None):
        super().__init__()
        self.encoder = BasicRecurrentEntityEncoder(embedding_matrix=embedding_matrix,
                                                   max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                                   entity_cell=entity_cell)
        self.decoder = RecurrentEntitiyDecoder(embedding_matrix=embedding_matrix,
                                               entity_cell=entity_cell, entity_embedding_dim=entity_embedding_dim,
                                               max_entity_num=max_entity_num, vocab_size=vocab_size,
                                               softmax_layer=softmax_layer, activation=activation)

    def call(self, inputs, keys, keys_mask, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True):
        """
         inputs=[prgrph, prgrph_mask, question]
        """
        prgrph, prgrph_mask, question = inputs
        entity_cell, entity_hiddens = self.encoder(inputs=[prgrph, prgrph_mask], keys=keys,
                                                   initial_entity_hidden_state=initial_entity_hidden_state,
                                                   use_shared_keys=use_shared_keys,
                                                   return_last=return_last)
        self.decoder.entity_cell=entity_cell
        output=self.decoder(inputs=[entity_hiddens,question],keys_mask=keys_mask)
        return output

