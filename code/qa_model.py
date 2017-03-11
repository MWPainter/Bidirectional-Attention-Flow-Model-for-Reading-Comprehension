from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score_ours

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


# the session is created in train.py but it is passed into all functions


class Encoder(object):
    def __init__(self, state_size, embedding_dim):
        self.state_size = size
        self.embedding_dim = vocab_dim # the dimension of the wor
        cell = rnn.cell.basicLSTMCell(self.size)
        
    def encode(self, question, question_mask, context_paragraph, context_mask):
        """
        Encoder function. Encodes the question and context paragraphs into some hidden representation.
        This function assumes that the question has been padded already to be of length FLAGS.question_max_length 
        and that the context paragraph has been padded to the length of FLAGS.context_paragraph_max_length

        :param question: A tf.placeholder for words in the question. 
                Dims = [batch_size, question_length, embedding_dimension]
        :param question_mask: A tf.placholder for a mask over the question. 
                Dims = [batch_size, question_length, 1]
                0 in mask indicates if the word is padding (<pad>), 1 otherwise
        :param context_paragraph: A tf.placeholder for words in the context paragraph 
                Dims = [batch_size, context_paragraph_length, embedding_dimension]
        :param context_mask: A tf.placholder for a mask over the context paragraph. 
                Dims = [batch_size, context_paragraph_length, 1]
                0 in mask indicates if the word is padding (<pad>), 1 otherwise
        :param var_scope: The tf variable scope to use
        :param reuse: boolean whether to reuse variables in this scope
        :return: An (tf op of) encoded representation of the input of the form (attention_vector, context_vectors)
                'attention_vector' is the attention vector for the sequence
                dims = [batch_size, state_size]
                'context_vectors' is the states of the (rolled out) rnn for the context paragraph
                dims = [batch_size, context_paragraph_length, state_size]
        """

        # Build a BiLSTM layer for the question (we only want the concatinated end vectors here)
        question_vector, _ = build_rnn(self.cell, question, question_mask, scope="question_BiLSTM", reuse=True)

        # Concatanate the question vector to every word in the context paragraph, by tiling the question vector and concatinating
        question_vector_tiled = tf.expand_dims(question_vector, 1)
        question_vector_tiled = tf.tile(question_vector_tiled, tf.pack([1, tf.shape(context_paragraph)[1], 1]))
        context_input = tf.concat(context_paragraph, question_vector_tiled)

        # Build BiLSTM layer for the context (want all output states here)
        _, context_states = build_rnn(context_input, context_mask, scope="context_BiLSTM", reuse=True)

        # Create attention vector
        attention_vector = create_attention_vector(context_states, question_vector, scope="AttentionVector", reuse=True)

        # Retuuuurn
        return (attention_vector, context_states)



    def build_rnn(inpt, mask, scope="default scope", reuse=False):
        """
        Helper function to build a rolled out rnn. We need the mask to tell tf how far to roll out the network
        
        :param inpt: input to the rnn, should be a tf.placeholder/tf.variable
            Dims: [batch_size, input_sequence_length, embedding_size], input_sequence_length is the question/context_paragraph length
        :param mask: a tf.variable for the mask of the input, if there is a 0 in this, then the corresponding word is a <pad>
            Dims: [batch_size, input_sequence_length]
        :param scope: the variable scope to use
        :param reuse: boolean if we should reuse parameters in each cell in the rolled out network (i.e. is it theoretically the "same cell" or different?)
        :return: (concat(fw_final_state, bw_final_state), concat(fw_all_states, bw_all_states)), 
                final_state is the final states of (RELEVANT part) of the states
                dim = [batch_size, cell_size] (cell_size = hidden state size of the lstm)
                so dim of concatinated state is [batch_size, 2*cell_size]
                all_states are the tf.variable's for all of the hidden states
                dim = [batch_size, input_sequence_length, cell_size]
                dim of the concatinated states is [batch_size, input_sequence_length, 2*cell_size]
        """

        # build the dynamic_rnn, compute lengths of the sentences (using mask) so tf knows how far it needs to roll out rnn
        # fw_outputs, bw_outputs is of dim [batch_size, input_sequence_length, embedding_size], n.b. "input_lengths" below is smaller than "input_sequence_length"
        # we think second output is the ("true") final state, but TF docs are ambiguous AF, so I don't really know. There may be problems here...
        with vs.variable_scope(scope, reuse):
            input_length = tf.reduce_sum(mask, axis=1) # dim = [batch_size]
            (fw_outputs, bw_outputs), (fw_final_state, bw_final_state) = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, sequence_length=input_length) 
            return (tf.concat([fw_final_state, bw_final_state], 1), tf.concat([fw_outputs, bw_outputs], 2))



    def create_attention_vector(rnn_states, cur_state, scope="default scope", reuse=False):
        """
        Helper function to create an attention vector. rnn_states and cur_state are vectors with dimension state_size
        rnn_states encorporates inputes of length 'seq_len'

        :param rnn_state: the states which an rnn went through, and we want to learn which are relevant
            dim = [batch_size, seq_len, state_size]
        :param cur_state: the current state which we want to attend to
            dim = [batch_size, state_size]       
        :param scope: the variable scope to use
        :param reuse: boolean if we should reuse parameters in each cell in the rolled out network (i.e. is it theoretically the "same cell" or different?)
        :return: an attention vector, which is a weighted combination of the rnn_states, encorporating relevant information from rnn_states
        """

        # Compute scores for each rnn state
        state_size = self.size * 2 # vectors we're working with are concatinations of two vectors
        batch_size = tf.shape(rnn_states)[0]
        seq_len = tf.shape(rnn_states)[1]
        with vs.variable_scope(scope, reuse):
            # Setup variables to be able to make the matrix product
            inner_product_matrix = tf.get_variable("inner_produce_matrix", shape=(state_size, state_size), initializer=tf.contrib.layers.xavier_initialization()) # dim = [statesize, statesize]
            inner_product_matrix_tiled = tf.expand_dims(tf.expand_dims(inner_product_matrix, 0), 0) # dim = [1, 1, statesize, statesize]
            inner_product_matrix_tiled = tf.tile(inner_product_matrix_tiled, tf.pack([batch_size, seq_len, 1, 1])) # dim = [batch_size, seq_len, statesize, statesize]
            cur_state_tiled = tf.expand_dims(tf.expand_dims(cur_state, 0), 0) # dim = [1, 1, statesize]
            cur_state_tiled = tf.tile(cur_state_tiled, tf.pack([batch_size, seq_len, 1])) # dim = [batch_size, seq_len, statesize]
            cur_state_tiled = tf.expand_dims(cur_state_tiled, 3) # dim = [batch_size, seq_len, state_size, 1]
            rnn_state_expanded = tf.expand_dims(rnn_states, 2) # dim = [batch_size, seq_len, 1, state_size]

            # Matrix product. Each input is a rank 4 tensor. For each index in batch_size, seq_len, we comupute an quadratic form. [1, state_size] * [state_size, state_size] * [state_size, 1]
            attention_scores = tf.matmul(tf.matmul(rnn_state_expanded, inner_product_matrix_tiled), cur_state_tiled) # dim = [batch_size, seq_len, 1, 1]
            attention_scores = tf.reduce_max(tf.reduce_max(attention_scores, axis=3), axis=2) # dim = [batch_size, seq_len], just used to reduce rank of tensor

            # Attention vector is attention scores run through softmax
            attention = tf.nn.softmax(attention_scores) # dim = [batch_size, seq_len]

            # Take a weighted sum over the vectors in the rnn, the multiply broadcasts appropriately (1 over state_size)
            attention_vector = tf.reduce_sum(tf.multiply(rnn_states, attention_vector), axis = 1) # before reduce sum dim = [batch_size, seq_len, state_size], after dim = [batch_size, state_size]
            return attention_vector


            


""" SHHHHHHIIIIIIEEEEET from coding session
        question_length = FLAGS.question_max_length
        context_paragraph_length = FLAGS.context_paragraph_length
        dropout_rate = FLAGS.dropout

        # TODO: get words from embeddings
        distributed_questions = # questions input, dimensions [batch_size, question_lenth, embedding_dim] (n.b. embedding_dim = self.vocab_dim in the code)
      
        with vs.variable_scope(scope, reuse): # if you change the scope, you'll be using two different self.cell's
            # build your dynamic_rnn in here
            mask = tf.sign(context_paragraph)
            context_len = tf.reduce_sum(mask, axis = 1)
            o, _ = dynamic_rnn(self.cell, inputs, srclen = srclen, inputs_state=None) # the first returned value is a 3d object with all the hidden states
            #(fw_o, bw_o), _ = bidirectional_dynamic_rnn(self.cell, self.cell, inputs) fw_o and bw_o are forward and backward things
            o = tf.concat(fw_o[:,-1,:], bw_o[:,0,:])
        return o # (N, T, d) N is the mini batch size, T is the length of the sentence or context, d is the dimension of each word
    # encoder = Encoder()
    # encoder.encode(question)
    # encoder.encode(paragrapgh, resue=True) this will use the same cell since reuse=True
    def  encode_w_attn(self, inputs, masks, prev_states, scope="", reuse = false):
        self.attn_cell = AttnGRUCell = ArrnCRUCell(self.size, prev_states)
        with vs.variable_scope(scope, reuse):
            0, _= dynamic_rnn(self.attn_Cell, inputs, srclem=srclen, initial_state=None)

class AttnGRUCell(rnn_cell.GRUCell):
    def __init__():
        # call super class
        return
    def __call__():
        # this is called when it is called
        return
"""

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """


        # Bardia's work:
        
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
        x = tf.concatenate(h_p, tf.tile(av, tf.pack([tf.shape(h_p)[0], 1])))
        with vs.scope("answer_start"):
            val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            a_s = rnn_cell._linear([val], output_size= self.output_size)
        with vs.scope("answer_end"):
            val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            a_e = rnn_cell._linear([val], output_size= self.output_size)
        return a_s, a_e
    
        """
        # h_q, h_p: 2-d TF variable
        with vs.scope("answer_start"):
            a_s = rnn_cell._linear([h_q, h_p], output_size= self.output_size)
        with vs.scope("answer_end"):
            a_e = rnn_cell._linear([h_q, h_p], output_size= self.output_size)
        # with vs.scope("linear_default"):
        #   tf.get_variable("W", shape = [])
        #   tf.get_variable("b", shape = [])
        #   h_q = W + b
        #   tf.get_variable("W", shape = [])
        #   tf.get_variable("b", shape = [])
        #   h_p = W + b
        # creates h_qW + b_q + h_pW + b_p
        return
        """
    
class QASystem(object):
    # i added embeddings down here, make sure you'll change it in train.py
    def __init__(self, encoder, decoder, embeddings, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        # is this the way to do it????
        # do we need to do reshape like we did in q2_rnn
        self.embeddings = tf.constant(tf.convert_to_tensor(embeddings, dtype = tf.float32), name = 'embedding')
        self.encoder = encoder
        self.decoder = decoder

        # start answer and end answer
        # we probably need to change output_size to max_context_length or something similar
        self.answer_start = tf.placeholder(shape = [None, self.FLAGS.output_size])
        self.answer_end = tf.placeholder(shape = [None, self.FLAGS.output_size])

        # question, paragraph (context), answer, and dropout placeholders
        self.question_placeholder = tf.placeholder(tf.int32, (None, FLAGS.question_max_length, FLAGS.embedding_size))
        self.context_placeholder = tf.placeholder(tf.int32, (None, FLAGS.context_paragraph_max_length, FLAGS.embedding_size))
        self.question_mask_placeholder = tf.placeholder(tf.boolean32, (None, FLAGS.question_paragraph_max_length))
        self.context_mask_placeholder = tf.placeholder(tf.boolean32, (None, FLAGS.context_paragraph_max_length))
        
        self.answer_placeholder = tf.placeholder(tf.int32, (None, 2))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        params = tf.trainable_variables()
        optimizer = FLAGS.optimizer
        self.updates = get_optimizer(optimizer)(FLAGS.learning_rate).minimize(self.loss)

        
        # define an optimization proedure
        # sess.run(self.updates???)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        
        attentionVector, contextVectors = self.encoder.encode(self.question_var, self.question_mask_var, self.paragraph_var, self.context_mask_var)
        self.a_s, self.a_e = self.decoder.decode(attentionVector, contextVectors)
        
        # q_o, q_h = encoder.encode(self.question_Var)
        # p_o, p_h= encoder.encode(self.paragraph_Var, init_state = q_h, reuse = True)
        # self.a_s, self.a_e = decoder.decode(q_h, p_h)

        
    def create_feed_dict(self, question_batch, context_batch, answer_start_batch = None, answer_end_batch = None, dropout = None):
        feed_dict = {}
        
        question_mask_batch = []
        for i in range(len(question_batch)):
            q = [True] * len(question_batch[i])
            for _ in range(FLAGS.question_max_length - len(question_batch[i])):
                question_batch[i].append(0)
                q.append(False)
            question_mask_batch.append(q)

        feed_dict[self.question_mask_placeholder] = question_mask_batch
        feed_dict[self.question_placeholder] = question_batch
        
        context_mask_batch = []
        for i in range(len(context_batch)):
            q = [True] * len(context_batch[i])
            for _ in range(FLAGS.context_paragraph_max_length - len(context_batch[i])):
                context_batch[i].append(0)
                q.append(False)
            context_mask_batch.append(q)
            
        feed_dict[self.context_mask_placeholder] = context_mask_batch
        feed_dict[self.context_placeholder] = context_batch
            
        if answer_start_batch is not None:
            feed_dict[self.answer_start] = answer_start_batch

        if answer_end_batch is not None:
            feed_dict[self.answer_end] = answer_end_batch

        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict                
            

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_s, self.answer_start)
            l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_e, self.answer_end)
            self.loss = l1 + l2
            

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            #glove_matrix = np.load()['glove']
            #params = tf.constant(glove_matrix) # if you wanna train the embeddings too, put it in a variable (inside the init function)
            self.question_var = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)
            self.context_var = tf.nn.embedding_lookup(self.embeddings, self.context_placeholder)
            self.question_mask_var = tf.nn.embedding_lookup(self.embeddings, self.question_mask_placeholder)
            self.context_mask_var = tf.nn.embedding_lookup(self.embeddings, self.context_mask_placeholder)

    # this function calls answer bellow and is called by train at the bottom of the page
    # returns the f1 and em scores of an epoch


    # check whether the for loop is necessary. feel like should work with the full batch
    def evaluate_answer(self, session, paragraphs, questions, answers_start, answers_end, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        ######### the f1_score and exact_match_score functions defined work with strings
        ######### need to write new ones that work with lists like below 
        
        f1 = 0.
        em = 0.

        for p, q, r1, r2 in zip(paragraphs, questions, answers_start, answers_end):
            a_s, a_e = self.answer(session, p, q)
            answer = p[a_s, a_e]
            f1 += f1_score_ours(answer, [r1, r2])
            em += exact_match_score(answer, [r1, r2])
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em
    

    # this function is only called by evaluate_answer above and calls decode below.
    # returns indices of the most probable start and end words
        
    def answer(self, session, paragraph, question):
        yp1, yp2 = self.decode(session, paragraph, question)

        a_s = np.argmax(yp1, axis = 1)
        a_e = np.argmax(yp2, axis = 1)
        return (a_s, a_e)
    
    # this function is only called by answer above. returns probabilities for the start and end words
    
    def decode(self, session, test_paragraph, test_question):

        input_feed = {}
        """
        could we just have the following?
        input_feed = create_feed_dict(test_paragraph, test_question)
        """
        input_feed[self.context_var] = test_paragraph
        input_feed[self.question_var] = test_question

        output_feed = [self.a_s, self.a_e]
        outputs = session.run(output_feed, feed_dict = input_feed)
        return outputs    


    def optimize(self, session, question_batch, context_batch, answer_start_batch, answer_end_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.create_feed_dict(question_batch, context_batch, answer_start_batch, answer_end_batch)

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x


        #grad_norm, param_norm
        output_feed = [self.updates, self.loss]

        outputs = session.run(output_feed, feed_dict = input_feed)

        return outputs[1]

    def train(self, session, train_dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        q_val, p_val, a_val_s, a_val_e = val_dataset
        for e in range(self.FLAGS.epochs):
            for q, p, a_s, a_e in train_dataset[e]:
                loss = self.optimize(session, q, p, a_s, a_e)
                # save your model here
                saver = tf.saver()
                saver.save(session, train_dir + "epoch_" + str(e))
                val_loss = self.validate(p_val, val_dataset)

                self.evaluate_answer(session, q_val, p_val, a_val_s, a_val_e, 100, True)
                self.evaluate_answer(session, q, p, a_s, a_e, 100, True) # doing this cuz we wanna make sure it at least works well for the stuff it's already seen
                        



    # the following function is called from train above and only calls the function test below

    def validate(self, sess, valid_dataset): # only used for unseen examples, ie when you wanna check your model
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """

        question, context, start_answer, end_answer = valid_dataset
        valid_cost = self.test(sess, question, context, start_answer, end_answer)

        return valid_cost

    
    # the following function is called from validate above
    
    def test(self, session, question, paragraph, answer_start, answer_end):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        input_feed = self.create_feed_dict(paragraph, question, answer_start, answer_end)
        """
        input_feed[self.context_var] = paragraph
        input_feed[self.question_var] = question
        input_feed[self.answer_start] = answer_start
        input_feed[self.answer_end] = answer_end
        """
        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed) # sessions always return real things
        return outputs



        
