import numpy as np
import tensorflow as tf
import Model
import pickle
from tensorflow.contrib import seq2seq


def calculate_loss(targets, outputs):
    """
    Args:
        inputs: outputs shape : [batch_size,max_sents_num*max_sents_len, vocab_size]
                lstm_targets shape : [batch_size, max_sents_num*max_sents_len]
                mask : [batch_size, max_sents_num*max_sents_len]

    """
    # one_hot_labels = tf.one_hot(lstm_targets, outputs.shape[1])
    # print('outpus shape:', outputs.shape, outputs)
    # print('one_hot_labels shape:', one_hot_labels.shape)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=outputs)
    # print('loss', loss)
    print('IN LOSS FUNCTION')
    print('outputs shape:', outputs.shape)
    print('output dtype:', outputs.dtype)
    print('targets dtype:', targets.dtype)
    targetss=(tf.one_hot(tf.squeeze(tf.cast(targets,tf.int32),axis=1),depth=30522)*(1-(10**-7))+10**-7)

    # print('mask dtype:', mask.dtype)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,labels=targetss)
    return loss


def train(prgrphs, prgrphs_mask, questions, answers, keys, keys_mask, embedding_matrix, max_entity_num,
          entity_embedding_dim, vocab_size, learning_rate, save_path, batch_size, validation_split, epochs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)
    model = Model.Model(embedding_matrix=embedding_matrix, max_entity_num=max_entity_num,
                        entity_embedding_dim=entity_embedding_dim,
                        vocab_size=vocab_size)
    # output=model([prgrphs[:2], prgrphs_mask[:2], questions[:2], keys[:2], keys_mask[:2]])
    # with tf.Session() as sess:
    #     print("in session")
    #     sess.run(tf.global_variables_initializer())
    #     output=sess.run(output)
    #     print(output)


    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # answerss=(tf.one_hot(tf.squeeze(tf.cast(answers,np.int32),axis=1),depth=30522)*(1-(10**-7))+10**-7)
    answerss=np.eye(30522)[np.squeeze(answers,axis=1)]
    history = model.fit(x=[prgrphs, prgrphs_mask, questions, keys, keys_mask], y=answerss, batch_size=batch_size,
                        validation_split=validation_split, epochs=epochs, callbacks=[cp_callback])

    return history

if __name__=="__main__":
    with open("q1_train_paragraphs.pkl", "rb") as file:
        paragraphs = pickle.load(file)
    with open("q1_train_paragraphs_mask.pkl", "rb") as file:
        paragraphs_mask = pickle.load(file)
    with open("q1_train_questions.pkl", "rb") as file:
        questions = pickle.load(file)
    with open("q1_train_answers.pkl", "rb") as file:
        answers = pickle.load(file)
    embedding_matrix = None
    with open('embeddings_table.pkl', 'rb') as file:
        embedding_matrix = pickle.load(file)
    print(embedding_matrix.shape[0])

    with open('config.pkl', 'rb') as file:
        config = pickle.load(file)
    paragraphs_num = config['paragraphs_num']
    max_sent_num = config['max_sent_num']
    max_sent_len = config['max_sent_len']
    max_ques_len = config['max_ques_num']

    keys = np.random.normal(size=[paragraphs_num, 20, 100])
    keys_mask = np.ones([paragraphs_num, 20],np.bool)
    # print(paragraphs_num)
    # print(paragraphs[-1], paragraphs_mask[-1])
    batch_size=32
    paragraphs=paragraphs[:paragraphs_num-(paragraphs_num % batch_size)]
    paragraphs_mask=paragraphs_mask[:paragraphs_num-(paragraphs_num % batch_size)]
    keys=keys[:paragraphs_num-(paragraphs_num % batch_size)]
    keys_mask=keys_mask[:paragraphs_num-(paragraphs_num % batch_size)]
    questions=questions[:paragraphs_num-(paragraphs_num % batch_size)]
    answers=answers[:paragraphs_num-(paragraphs_num % batch_size)]
    train(paragraphs, paragraphs_mask, questions, answers, keys, keys_mask, embedding_matrix, 20, 100, 30522,
          0.01, "/content/drive/My Drive/Colab Notebooks/bAbI", batch_size, 0, 5)
