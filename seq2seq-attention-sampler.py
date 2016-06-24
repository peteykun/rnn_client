
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import os, time, sys

# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Run a repairing seq2seq RNN.')
parser.add_argument('task_name', help='The task to run.')
parser.add_argument('batch_size', type=int)
parser.add_argument('embedding_dim', type=int)
parser.add_argument('memory_dim', type=int)
parser.add_argument('num_layers', type=int)
parser.add_argument('resume_at', type=int)
parser.add_argument('rnn_cell', help='One of RNN, LSTM or GRU.')
parser.add_argument('ckpt_every', help='How often to checkpoint', type=int)
parser.add_argument("--correct_pairs", help="Include correct-correct pairs.")
parser.add_argument("--mux_network", help="Use the mux network to learn identifiers.")

args = parser.parse_args()

# In[2]:

task_name    = args.task_name

data_folder  = task_name + '_data'
ckpt_folder  = task_name + '_checkpoints'

# Options
shuffle_data  = True                # Shuffles the input data if set to True
correct_pairs = args.correct_pairs  # Removes correct-correct pairs if set to False
mux_network   = args.mux_network    # Use a MUX network for learning IDs if set to True
ckpt_every    = args.ckpt_every     # Store checkpoints every `ckpt_every` steps


# In[18]:

# Configurable data variables
batch_size = args.batch_size
embedding_dim = args.embedding_dim

# Network hyperparameters
memory_dim = args.memory_dim
num_layers = args.num_layers

# Resume training from saved model? (0 = don't use saved model)
resume_at = args.resume_at

# Cell type: one of RNN, LSTM and GRU
rnn_cell = args.rnn_cell


# In[4]:

# Load data
train_x = np.load(os.path.join(data_folder, 'mutated-train.npy'))
train_y = np.load(os.path.join(data_folder, 'fixes-train.npy'))
train_s = np.load(os.path.join(data_folder, 'select-train.npy'))

valid_x = np.load(os.path.join(data_folder, 'mutated-validation.npy'))
valid_y = np.load(os.path.join(data_folder, 'fixes-validation.npy'))
valid_s = np.load(os.path.join(data_folder, 'select-validation.npy'))

test_x = np.load(os.path.join(data_folder, 'mutated-test.npy'))
test_y = np.load(os.path.join(data_folder, 'fixes-test.npy'))
test_s = np.load(os.path.join(data_folder, 'select-test.npy'))

tl_dict = np.load(os.path.join(data_folder, 'translate_dict.npy')).item()

# Shuffle if required
if shuffle_data:
    # Check to see if shuffled data is available
    try:
        train_x = np.load(os.path.join(data_folder, 'shuffled/mutated-train.npy'))
        train_y = np.load(os.path.join(data_folder, 'shuffled/fixes-train.npy'))
        train_s = np.load(os.path.join(data_folder, 'shuffled/select-train.npy'))

        valid_x = np.load(os.path.join(data_folder, 'shuffled/mutated-validation.npy'))
        valid_y = np.load(os.path.join(data_folder, 'shuffled/fixes-validation.npy'))
        valid_s = np.load(os.path.join(data_folder, 'shuffled/select-validation.npy'))

        test_x = np.load(os.path.join(data_folder, 'shuffled/mutated-test.npy'))
        test_y = np.load(os.path.join(data_folder, 'shuffled/fixes-test.npy'))
        test_s = np.load(os.path.join(data_folder, 'shuffled/select-test.npy'))
        
        print "Successfully loaded shuffled data."
        sys.stdout.flush()
    
    # If not generate it
    except IOError:
        print "Generating shuffled data..."
        sys.stdout.flush()
        
        # Shuffle
        triples = zip(list(train_x), list(train_y), list(train_s))
        np.random.shuffle(triples)
        train_x, train_y, train_s = zip(*triples)

        triples = zip(list(valid_x), list(valid_y), list(valid_s))
        np.random.shuffle(triples)
        valid_x, valid_y, valid_s = zip(*triples)

        triples = zip(list(test_x), list(test_y), list(test_s))
        np.random.shuffle(triples)
        test_x, test_y, test_s = zip(*triples)

        # Convert to np array
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_s = np.array(train_s)

        valid_x = np.array(valid_x)
        valid_y = np.array(valid_y)
        valid_s = np.array(valid_s)

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_s = np.array(test_s)
    
        # Save for later
        try:
            os.mkdir(os.path.join(data_folder, 'shuffled'))
        except OSError:
            pass
        
        np.save(os.path.join(data_folder, 'shuffled/mutated-train.npy'), train_x)
        np.save(os.path.join(data_folder, 'shuffled/fixes-train.npy'), train_y)
        np.save(os.path.join(data_folder, 'shuffled/select-train.npy'), train_s)
        
        np.save(os.path.join(data_folder, 'shuffled/mutated-validation.npy'), valid_x)
        np.save(os.path.join(data_folder, 'shuffled/fixes-validation.npy'), valid_y)
        np.save(os.path.join(data_folder, 'shuffled/select-validation.npy'), valid_s)
        
        np.save(os.path.join(data_folder, 'shuffled/mutated-test.npy'), test_x)
        np.save(os.path.join(data_folder, 'shuffled/fixes-test.npy'), test_y)
        np.save(os.path.join(data_folder, 'shuffled/select-test.npy'), test_s)

if not correct_pairs:
    # Remove correct-correct pairs    
    new_train_x = []
    new_train_y = []
    new_train_s = []
    
    new_valid_x = []
    new_valid_y = []
    new_valid_s = []
    
    new_test_x = []
    new_test_y = []
    new_test_s = []
    
    for i in range(len(train_x)):
        if train_y[i][1] != 0:
            new_train_x.append(train_x[i])
            new_train_y.append(train_y[i])
            new_train_s.append(train_s[i])
            
    for i in range(len(valid_x)):
        if valid_y[i][1] != 0:
            new_valid_x.append(valid_x[i])
            new_valid_y.append(valid_y[i])
            new_valid_s.append(valid_s[i])
            
    for i in range(len(test_x)):
        if test_y[i][1] != 0:
            new_test_x.append(test_x[i])
            new_test_y.append(test_y[i])
            new_test_s.append(test_s[i])
            
    # Convert to np array
    train_x = np.array(new_train_x)
    train_y = np.array(new_train_y)
    train_s = np.array(new_train_s)

    valid_x = np.array(new_valid_x)
    valid_y = np.array(new_valid_y)
    valid_s = np.array(new_valid_s)

    test_x = np.array(new_test_x)
    test_y = np.array(new_test_y)
    test_s = np.array(new_test_s)
else:
    print "Including correct (i.e. no fix required) examples..."
    sys.stdout.flush()
    
if not mux_network:
    # Discard select line values of 1 (are 'name' types)
    new_train_x = []
    new_train_y = []
    new_train_s = []
    
    new_valid_x = []
    new_valid_y = []
    new_valid_s = []
    
    new_test_x = []
    new_test_y = []
    new_test_s = []
    
    for i in range(len(train_x)):
        if train_s[i] != 1:
            new_train_x.append(train_x[i])
            new_train_y.append(train_y[i])
            new_train_s.append(train_s[i])
            
    for i in range(len(valid_x)):
        if valid_s[i] != 1:
            new_valid_x.append(valid_x[i])
            new_valid_y.append(valid_y[i])
            new_valid_s.append(valid_s[i])
            
    for i in range(len(test_x)):
        if test_s[i] != 1:
            new_test_x.append(test_x[i])
            new_test_y.append(test_y[i])
            new_test_s.append(test_s[i])
            
    # Convert to np array
    train_x = np.array(new_train_x)
    train_y = np.array(new_train_y)
    train_s = np.array(new_train_s)

    valid_x = np.array(new_valid_x)
    valid_y = np.array(new_valid_y)
    valid_s = np.array(new_valid_s)

    test_x = np.array(new_test_x)
    test_y = np.array(new_test_y)
    test_s = np.array(new_test_s)
else:
    print "Using a MUX network..."
    sys.stdout.flush()

# Obtain counts
assert(len(train_x) == len(train_y))
assert(len(valid_x) == len(valid_y))
assert(len(test_x)  == len(test_y))

assert(len(train_x) == len(train_s))
assert(len(valid_x) == len(valid_s))
assert(len(test_x)  == len(test_s))

num_train = len(train_x)
print 'Training:', num_train, 'examples'
sys.stdout.flush()

num_validation = len(valid_x)
print 'Validation:', num_validation, 'examples'
sys.stdout.flush()

num_test = len(test_x)
print 'Test:', num_test, 'examples'
sys.stdout.flush()

# Make sure all shapes are A-OK
assert(len(np.shape(train_x)) == 2)
assert(len(np.shape(train_y)) == 2)
assert(len(np.shape(train_s)) == 1)

assert(len(np.shape(valid_x)) == 2)
assert(len(np.shape(valid_y)) == 2)
assert(len(np.shape(valid_s)) == 1)

assert(len(np.shape(test_x))  == 2)
assert(len(np.shape(test_y))  == 2)
assert(len(np.shape(test_s))  == 1)

# In[5]:

# Infer some data variables
seq_length = np.shape(train_x)[1]
out_seq_length = np.shape(train_y)[1]
vocab_size = len(tl_dict) # Includes a 0 for padding

print 'In sequence length:', seq_length
sys.stdout.flush()
print 'Out sequence length:', out_seq_length
sys.stdout.flush()
print 'Vocabulary size:', vocab_size
sys.stdout.flush()


# In[6]:

# Don't use all the VRAM!
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# First build input placeholders and constants. The `seq2seq` API generally deals with lists of tensors, where each tensor represents a single timestep. An input to an embedding encoder, for example, would be a list of `seq_length` tensors, each of which is of dimension `batch_size` (specifying the embedding indices to input at a particular timestep).
# 
# We allocate a `labels` placeholder using the same convention. A `weights` constant specifies cross-entropy weights for each label at each timestep.

# In[7]:

if mux_network:
    enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(seq_length + 1)]
else:
    enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in range(out_seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the decoder output
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")] +
           [tf.placeholder(tf.int32, shape=(None,), name="dec_inp%i" % t) for t in range(out_seq_length - 1)])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))


# Build the sequence-to-sequence graph.
# 
# There is a **lot** of complexity hidden in these two calls, and it's certainly worth digging into both in order to really understand how this is working.

# In[8]:

if rnn_cell == 'LSTM':
    constituent_cell = tf.nn.rnn_cell.BasicLSTMCell(memory_dim)
elif rnn_cell == 'GRU':
    constituent_cell = tf.nn.rnn_cell.GRUCell(memory_dim)
elif rnn_cell == 'RNN':
    constituent_cell = tf.nn.rnn_cell.BasicRNNCell(memory_dim)
else:
    raise Exception('unsupported rnn cell type: %s' % rnn_cell)

if num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([constituent_cell] * num_layers)
else:
    cell = constituent_cell

# Without teacher forcing, with attention
dec_outputs, dec_memory = tf.nn.seq2seq.embedding_attention_seq2seq(enc_inp, dec_inp, cell, vocab_size+1, vocab_size+1, embedding_dim, feed_previous=True)


# Build a standard sequence loss function: mean cross-entropy over each item of each sequence.

# In[9]:

loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size + 1)


# Build an optimizer.

# In[10]:

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)


# In[11]:

saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)


# # Restore variables
# Optionally restore variables

# In[13]:

if resume_at > 0:
    saver.restore(sess, os.path.join(ckpt_folder, 'saved-model-attn-' + str(resume_at)))


# In[14]:

if resume_at == 0:
    sess.run(tf.initialize_all_variables())


# # Helper Methods

# In[15]:

def line_equal(y, y_hat):
    tilde_token = tl_dict['~']
    
    y_line = []
    y_hat_line = []
    
    for token in y:
        if token != tilde_token:
            y_line.append(token)
        else:
            break
    
    for token in y_hat:
        if token != tilde_token:
            y_hat_line.append(token)
        else:
            break

    return np.array_equal(y_line, y_hat_line)

def validate_batch(batch_id):
    X = valid_x[batch_id*batch_size:(batch_id+1)*batch_size]
    Y = valid_y[batch_id*batch_size:(batch_id+1)*batch_size]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(out_seq_length)})
    feed_dict.update({dec_inp[t]: Y[t] for t in range(out_seq_length - 1)})
        
    if mux_network:
        S = valid_s[batch_id*batch_size:(batch_id+1)*batch_size]
        feed_dict.update({enc_inp[seq_length]: S})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    dec_outputs_batch = sess.run(dec_outputs, feed_dict)
    Y_hat = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
    token_accuracy = float(np.count_nonzero(np.equal(Y, Y_hat)))/np.prod(np.shape(Y))

    Y = np.array(Y, dtype=np.int32).T
    Y_hat = np.array(Y_hat, dtype=np.int32).T

    repair_accuracy = 0
    localization_accuracy = 0

    for y, y_hat in zip(Y, Y_hat):
        if np.array_equal(y, y_hat):
            repair_accuracy += 1
            localization_accuracy += 1
        elif line_equal(y, y_hat):
            localization_accuracy += 1

    repair_accuracy = float(repair_accuracy)/float(batch_size)
    localization_accuracy = float(localization_accuracy)/float(batch_size)
        
    return loss_t, token_accuracy, repair_accuracy, localization_accuracy


def test_batch(batch_id):
    X = test_x[batch_id*batch_size:(batch_id+1)*batch_size]
    Y = test_y[batch_id*batch_size:(batch_id+1)*batch_size]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(out_seq_length)})
    feed_dict.update({dec_inp[t]: Y[t] for t in range(out_seq_length - 1)})
        
    if mux_network:
        S = test_s[batch_id*batch_size:(batch_id+1)*batch_size]
        feed_dict.update({enc_inp[seq_length]: S})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    dec_outputs_batch = sess.run(dec_outputs, feed_dict)
    Y_hat = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
    token_accuracy = float(np.count_nonzero(np.equal(Y, Y_hat)))/np.prod(np.shape(Y))

    Y = np.array(Y, dtype=np.int32).T
    Y_hat = np.array(Y_hat, dtype=np.int32).T

    repair_accuracy = 0
    localization_accuracy = 0

    for y, y_hat in zip(Y, Y_hat):
        if np.array_equal(y, y_hat):
            repair_accuracy += 1
            localization_accuracy += 1
        elif line_equal(y, y_hat):
            localization_accuracy += 1

    repair_accuracy = float(repair_accuracy)/float(batch_size)
    localization_accuracy = float(localization_accuracy)/float(batch_size)
        
    return loss_t, token_accuracy, repair_accuracy, localization_accuracy


# In[16]:

def train_batch(batch_id):
    X = train_x[batch_id*batch_size:(batch_id+1)*batch_size]
    Y = train_y[batch_id*batch_size:(batch_id+1)*batch_size]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T
        
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(out_seq_length)})
    feed_dict.update({dec_inp[t]: Y[t] for t in range(out_seq_length - 1)})
    
    if mux_network:
        S = train_s[batch_id*batch_size:(batch_id+1)*batch_size]
        feed_dict.update({enc_inp[seq_length]: S})

    _, loss_t = sess.run([train_op, loss], feed_dict)
        
    return loss_t


# # Train
# 
# Do not initialize variables if restoring from a saved file.  
# **Warning:** epoch numbers start from 0, and *will* overwrite your old saves!

# In[ ]:

# Validation
valid_loss   = []
valid_token  = []
valid_local  = []
valid_repair = []

for i in range(num_validation/batch_size):
    f_loss, f_token, f_repair, f_local = validate_batch(i)

    valid_loss.append(f_loss)
    valid_token.append(f_token)
    valid_local.append(f_local)
    valid_repair.append(f_repair)

    print "Loss: %g Token: %g Localization: %g Repair: %g" % (f_loss, f_token, f_local, f_repair)
    sys.stdout.flush()
    
valid_loss   = np.mean(valid_loss)
valid_token  = np.mean(valid_token)
valid_local  = np.mean(valid_local)
valid_repair = np.mean(valid_repair)

# Print validation information
print "[Validation] Loss: %g Token: %g Localization: %g Repair: %g" % (valid_loss, valid_token, valid_local, valid_repair)
sys.stdout.flush()

# Testing
test_loss   = []
test_token  = []
test_local  = []
test_repair = []

for i in range(num_test/batch_size):
    f_loss, f_token, f_repair, f_local = test_batch(i)

    test_loss.append(f_loss)
    test_token.append(f_token)
    test_local.append(f_local)
    test_repair.append(f_repair)

    print "Loss: %g Token: %g Localization: %g Repair: %g" % (f_loss, f_token, f_local, f_repair)
    sys.stdout.flush()
    
test_loss   = np.mean(test_loss)
test_token  = np.mean(test_token)
test_local  = np.mean(test_local)
test_repair = np.mean(test_repair)

# Print test information
print "[Test] Loss: %g Token: %g Localization: %g Repair: %g" % (test_loss, test_token, test_local, test_repair)
sys.stdout.flush()


# In[ ]:

sess.close()

