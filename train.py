import numpy as np
import tensorflow as tf
import Model


def train(prgrphs, prgrphs_mask, questions, answers, keys, keys_mask, embedding_matrix, max_entity_num,
          entity_embedding_dim, vocab_size, learning_rate, save_path, batch_size, validation_split, epochs):
    model = Model.Model(embedding_matrix=embedding_matrix, max_entity_num=max_entity_num,
                        entity_embedding_dim=entity_embedding_dim,
                        vocab_size=vocab_size)

    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(x=[[prgrphs, prgrphs_mask, questions], keys, keys_mask], y=answers, batch_size=batch_size,
                        validation_split=validation_split, epochs=epochs, callbacks=[cp_callback])

    return history

