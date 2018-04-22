# -*- coding:utf-8 -*-
__author__ = 'chenjun'

import tensorflow as tf
from tensorflow_models.seq2seq import Seq2SeqTranslate
from utils.util import *
from utils.translate_data_process import load_data_for_tensorflow
import time


def build_model(vocab_size_src=7752, vocab_size_tar=7784, hidden_size=128, layers_num=1, batch_size=400, n_epochs=2, evaluate=1):
    encoder_inputs, encoder_length, decoder_inputs, decoder_target = load_data_for_tensorflow("data2/train_data.pkl")
    num_batchs = encoder_inputs.shape[0] // batch_size
    print ("train_batch_num: %d" % (num_batchs))
    emb_src = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size_src, hidden_size))
    emb_tar = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size_tar, hidden_size))
    model = Seq2SeqTranslate(emb_src, emb_tar, vocab_size_src, vocab_size_tar, hidden_size, MAX_LENGTH, layers_num, learning_rate=0.01)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    epoch = 0
    start_time = time.time()
    total_batches = n_epochs * num_batchs
    cur_batches = 0
    while epoch < n_epochs:
        epoch += 1
        for batch_index in range(num_batchs):
            cur_batches += 1
            ei = encoder_inputs[batch_index * batch_size: (batch_index + 1) * batch_size]
            el = encoder_length[batch_index * batch_size: (batch_index + 1) * batch_size]
            di = decoder_inputs[batch_index * batch_size: (batch_index + 1) * batch_size]
            dt = decoder_target[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model.train(sess, ei, el, di, dt)
            if batch_index % 10 == 1:
                print ("epoch: %d/%d, batch: %d/%d, loss: %f" % (epoch, n_epochs, batch_index, num_batchs, loss))
                cur_time = time.time()
                cost_time = int(cur_time - start_time)
                total_time = cost_time * total_batches // cur_batches
                print ("cost time: %s/%s" % (second2time(cost_time), second2time(total_time)))
        # save model
        saver.save(sess, "./model/model", global_step=epoch)
    # evaluate
    for i in range(evaluate):
        index = np.random.randint(low=0, high=encoder_inputs.shape[0])
        ei = encoder_inputs[index]
        el = encoder_length[index]
        dt = decoder_target[index]
        generate = model.generate(sess, ei, el)
        print ("> ", indices2sentence(parse_output(ei), 'data2/fra_i2w.json'))
        print ("= ", indices2sentence(parse_output(dt), 'data2/eng_i2w.json'))
        print ("< ", indices2sentence(parse_output(generate), 'data2/eng_i2w.json'))
        print ("")

def load_model(evaluate=20):
    # load data
    tf.reset_default_graph()
    data_encoder_inputs, data_encoder_length, data_decoder_inputs, data_decoder_target = load_data_for_tensorflow("data2/train_data.pkl")
    with tf.Session() as sess:
        # load
        ckpt = tf.train.latest_checkpoint("./model/")
        saver = tf.train.import_meta_graph(ckpt+".meta")
        saver.restore(sess, ckpt)
        graph = tf.get_default_graph()
        model_encoder_inputs = graph.get_tensor_by_name("place_holder/encoder_inputs:0")
        model_encoder_length = graph.get_tensor_by_name("place_holder/encoder_length:0")
        model_decoder_inputs = graph.get_tensor_by_name("place_holder/decoder_inputs:0")
        generate_outputs = graph.get_tensor_by_name("seq2seq-generate/generate_outputs:0")
        # evaluate
        for i in range(evaluate):
            index = np.random.randint(low=0, high=data_encoder_inputs.shape[0])
            ei = data_encoder_inputs[index]
            el = data_encoder_length[index]
            dt = data_decoder_target[index]
            predict_decoder_inputs = np.asarray([[SOS_token] * MAX_LENGTH], dtype="int64")
            if data_encoder_inputs.ndim == 1:
                data_encoder_inputs = data_encoder_inputs.reshape((1, -1))
                data_encoder_length = data_encoder_length.reshape((1,))
            res = [generate_outputs]
            generate = sess.run(res,
                                feed_dict={model_encoder_inputs: data_encoder_inputs,
                                           model_decoder_inputs: predict_decoder_inputs,
                                           model_encoder_length: data_encoder_length
                                           })[0]
            print("> ", indices2sentence(parse_output(ei), 'data2/fra_i2w.json'))
            print("= ", indices2sentence(parse_output(dt), 'data2/eng_i2w.json'))
            print("< ", indices2sentence(parse_output(generate), 'data2/eng_i2w.json'))
            print("")


if __name__ == '__main__':
    build_model()
    load_model()