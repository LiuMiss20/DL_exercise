
# coding: utf-8

# # TV Script Generation

import helper
import problem_unittests as tests


data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)


# ## Explore the Data   ##########################################
view_sentence_range = (0, 10)  #view different parts of the data.
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Preprocessing  ##########################################
# 【第一部分】 Lookup Table
from collections import Counter
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_count = Counter(text)
    #sorted_word = sorted(word_count, key=word_count.get, reverse=True) # key=word_count.get 按照key原始顺序排序，reverse=True 降序
    int_to_vocab = { idx:word for idx,word in enumerate(word_count)}
    vocab_to_int = { word:idx for idx,word in enumerate(word_count)}
    return vocab_to_int, int_to_vocab

tests.test_create_lookup_tables(create_lookup_tables)


# 【第二部分】 Tokenize Punctuation
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    Tokenize = {'.': '||Period||',
                '.': '||Period||',
                ',': '||Comma||',
                '"': '||Quotation_Mark||',    
                ';': '||Semicolon||',         
                '!': '||Exclamation_mark||',  
                '?': '||Question_mark||', 
                '(': '||Left_Parentheses||', 
                ')': '||Right_Parentheses||', 
                '--': '||Dash||',
                '\n': '||Return||'}   
    
    return Tokenize

tests.test_tokenize(token_lookup)


# 【第三部分】 Preprocess all the data and save it
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# 【第四部分】 Check Point
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# ## Build the Neural Network ######################################

# 【第一部分】 Check the Version of TensorFlow and Access to GPU
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# 【第二部分】 Input
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None,None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    return (inputs, targets, lr)
tests.test_get_inputs(get_inputs)


# 【第三部分】 Build RNN Cell and Initialize
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """

    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size) #?????????????????cell_hidden_size = state_size
    #cell = tf.contrib.run.DropoutWrapper(cell, output_keep_prob=keep_prob)
    num_of_layers = 3
    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_of_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    
    return (cell, initial_state)

tests.test_get_init_cell(get_init_cell)


# 【第四部分】 Word Embedding
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size,embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    #print ("embed_dim: ",embed_dim)  # 向量表达维度为 256
    #print ("input_data.shape: ",input_data.shape)  # (50, 5)
    #print ("embed.shap: ", embed.shape)   # word 的向量表达 ==特征 (50, 5, 256) ==(batch_size, num_step, embed_dim)
    return embed   # 返回input的向量表达

tests.test_get_embed(get_embed)


# 【第五部分】 Build RNN
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    #_,initial_state = get_init_cell(batch_size, rnn_size)
    
    output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')

    return (output, final_state)

tests.test_build_rnn(build_rnn)


# 【第六部分】 Build the Neural Network
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)      
    output, final_state = build_rnn(cell, embed)
    
    logits = tf.contrib.layers.fully_connected(output, vocab_size, activation_fn=None)
    #final_state = tf.identity(final_state, name='final_state')    
    return logits, final_state

tests.test_build_nn(build_nn)


# 【第七部分】 Batches
# For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
# ```
# [
#   # First Batch
#   [
#     # Batch of Input
#     [[ 1  2], [ 7  8], [13 14]]
#     # Batch of targets
#     [[ 2  3], [ 8  9], [14 15]]
#   ]
# 
#   # Second Batch
#   [
#     # Batch of Input
#     [[ 3  4], [ 9 10], [15 16]]
#     # Batch of targets
#     [[ 4  5], [10 11], [16 17]]
#   ]
# 
#   # Third Batch
#   [
#     # Batch of Input
#     [[ 5  6], [11 12], [17 18]]
#     # Batch of targets
#     [[ 6  7], [12 13], [18  1]]
#   ]
# ]
# ```
# 
# Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text) // (batch_size * seq_length)
    len_int_text = n_batches * (batch_size*seq_length)
    
    x = np.array(int_text[: len_int_text])
    y = np.hstack((np.array(int_text[1: len_int_text]) , np.array(int_text[0])))  #np.hstack()水平合并
    
    x_batches = np.split(x.reshape(batch_size, -1), n_batches, -1)
    y_batches = np.split(y.reshape(batch_size, -1), n_batches, -1)
    
    all_batches= np.array(list(zip(x_batches, y_batches)))
    return all_batches

tests.test_get_batches(get_batches)


# ## Neural Network Training  ############################################

# 【第一部分】 Hyperparameters
# Number of Epochs
num_epochs = 1000   # 600左右loss约为0.3左右 后续基本维持在0.29 / 0.3
# Batch Size
batch_size = 512  # depends on GPU memory usually，越大越好
# RNN Size  # number of units in the hidden layers
rnn_size = 512  #large enough to fit the data well. Again, no real “best” value.
# Embedding Dimension Size
embed_dim = 300  # 
# Sequence Length
seq_length = 11   # 每个句子平均的单词数  Should match the structure of the data||the size of the length of sentences you want to generate.
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 6   

save_dir = './save'


# 【第二部分】 Build the Graph
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# 【第三部分】 Train
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# 【第四部分】 Save Parameters
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))


# 【第五部分】 Checkpoint
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


# ##  Generate TV  Script  #######################################################
# 【第一部分】 Get Tensors
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    g = loaded_graph
    InputTensor = g.get_tensor_by_name("input:0")
    InitialStateTensor = g.get_tensor_by_name("initial_state:0")
    FinalStateTensor = g.get_tensor_by_name("final_state:0")    
    ProbsTensor = g.get_tensor_by_name("probs:0")

    return  InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor

tests.test_get_tensors(get_tensors)


# 【第二部分】 Choose Word
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # return int_to_vocab[probabilities.argmax()] # 不建议用，需一定的随机性
   
    word = [int_to_vocab[i] for i in range(len(int_to_vocab))]
    pred_word = np.random.choice(word, 3, p=probabilities)[0]  # [0] 返回第0个值

    return pred_word

tests.test_pick_word(pick_word)


# 【第三部分】 Generate TV Script
gen_length = 500
prime_word = 'moe_szyslak'  # 可选范围  homer_simpson, moe_szyslak, or Barney_Gumble

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)


# # The TV Script is Nonsensical
# It's ok if the TV script doesn't make any sense.  
# We trained on less than a megabyte of text.  
# In order to get good results, you'll have to use 【a smaller vocabulary】 or 【get more data】. 
# Luckly there's more data!  
# As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data). 

