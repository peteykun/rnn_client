
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import os, time, sys
from shutil import copyfile

# Parse command line arguments
import argparse

# Set up logging to a file
class Logger(object):
    def __init__(self, filename='log'):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

parser = argparse.ArgumentParser(description='Run a repairing seq2seq RNN.')
parser.add_argument('task_name', help='The task to run.')
parser.add_argument('batch_size', type=int)
parser.add_argument('embedding_dim', type=int)
parser.add_argument('memory_dim', type=int)
parser.add_argument('num_layers', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('resume_at', type=int)
parser.add_argument('resume_epoch', type=int)
parser.add_argument('resume_training_minibatch', type=int)
parser.add_argument('rnn_cell', help='One of RNN, LSTM or GRU.')
parser.add_argument('ckpt_every', help='How often to checkpoint', type=int)
parser.add_argument('dropout', help='Probability to use for dropout', type=float)
parser.add_argument('--correct_pairs', action="store_true", help="Include correct-correct pairs.")
parser.add_argument('--mux_network', action="store_true", help="Use the mux network to learn identifiers.")
parser.add_argument('--skip_training', action="store_true", help="Don't train, just validate and test at the specified checkpoint.")

args = parser.parse_args()

# In[2]:

task_name    = args.task_name
sys.stdout   = Logger(args.task_name)

data_folder  = task_name + '_data'
ckpt_folder  = task_name + '_checkpoints'

# Make checkpoint directories
try:
    os.mkdir(ckpt_folder)
except OSError:
    pass

try:
    os.mkdir(os.path.join(ckpt_folder, 'best'))
except OSError:
    pass

# Options
shuffle_data  = True                # Shuffles the input data if set to True
correct_pairs = args.correct_pairs  # Removes correct-correct pairs if set to False
mux_network   = args.mux_network    # Use a MUX network for learning IDs if set to True
ckpt_every    = args.ckpt_every     # Store checkpoints every `ckpt_every` steps

print 'Shuffle data:', shuffle_data
print 'Correct pairs:', correct_pairs
print 'Mux network:', mux_network
print 'Ckpt every:', ckpt_every

# In[18]:

# Configurable data variables
batch_size = args.batch_size
embedding_dim = args.embedding_dim

print 'Batch size:', batch_size
print 'Embedding dim:', embedding_dim

# Network hyperparameters
memory_dim = args.memory_dim
num_layers = args.num_layers

print 'Memory dim:', memory_dim
print 'Num layers:', num_layers

# Training variables
epochs = args.epochs

print 'Epochs:', epochs

# Resume training from saved model? (0 = don't use saved model)
resume_at = args.resume_at
resume_epoch = args.resume_epoch
resume_training_minibatch = args.resume_training_minibatch

print 'Resume at:', resume_at
print 'Resume epoch:', resume_epoch
print 'Resume training minibatch:', resume_training_minibatch

# Cell type: one of RNN, LSTM and GRU
rnn_cell = args.rnn_cell

print 'RNN cell:', rnn_cell

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

test_data_present = True

if len(test_x) == 0:
    test_data_present = False


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

        if test_data_present:
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

        if test_data_present:
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

        if test_data_present:
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

print 'after loading'

num_train = len(train_x)
print 'Training:', num_train, 'examples'
sys.stdout.flush()

num_validation = len(valid_x)
print 'Validation:', num_validation, 'examples'
sys.stdout.flush()

num_test = len(test_x)
print 'Test:', num_test, 'examples'
sys.stdout.flush()

if not correct_pairs:
    # Remove correct-correct pairs    
    new_train_x = []
    new_train_y = []
    new_train_s = []
    
    new_valid_x = []
    new_valid_y = []
    new_valid_s = []
    
    if test_data_present:
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

    if test_data_present:
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

    if test_data_present:
        test_x = np.array(new_test_x)
        test_y = np.array(new_test_y)
        test_s = np.array(new_test_s)
else:
    print "Including correct (i.e. no fix required) examples..."
    sys.stdout.flush()

print 'correct-correct pairs'

num_train = len(train_x)
print 'Training:', num_train, 'examples'
sys.stdout.flush()

num_validation = len(valid_x)
print 'Validation:', num_validation, 'examples'
sys.stdout.flush()

num_test = len(test_x)
print 'Test:', num_test, 'examples'
sys.stdout.flush()

    
if not mux_network:
    # Discard select line values of 1 (are 'name' types)
    new_train_x = []
    new_train_y = []
    new_train_s = []
    
    new_valid_x = []
    new_valid_y = []
    new_valid_s = []
    
    if test_data_present:
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
    
    if test_data_present:
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

    if test_data_present:
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

if test_data_present:
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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# First build input placeholders and constants. The `seq2seq` API generally deals with lists of tensors, where each tensor represents a single timestep. An input to an embedding encoder, for example, would be a list of `seq_length` tensors, each of which is of dimension `batch_size` (specifying the embedding indices to input at a particular timestep).
# 
# We allocate a `labels` placeholder using the same convention. A `weights` constant specifies cross-entropy weights for each label at each timestep.

# In[7]:

if args.dropout != 0:
    keep_prob = tf.placeholder(tf.float32)

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

if args.dropout != 0:
    constituent_cell = tf.nn.rnn_cell.DropoutWrapper(constituent_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)

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
#train_op = optimizer.minimize(loss)

gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)


# In[11]:

saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)
best_saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)


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
        
    if mux_network:
        S = valid_s[batch_id*batch_size:(batch_id+1)*batch_size]
        feed_dict.update({enc_inp[seq_length]: S})
    
    if args.dropout != 0:
        feed_dict.update({keep_prob = 1.0})

    loss_t = sess.run([loss], feed_dict)
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
        
    if mux_network:
        S = test_s[batch_id*batch_size:(batch_id+1)*batch_size]
        feed_dict.update({enc_inp[seq_length]: S})

    if args.dropout != 0:
        feed_dict.update({keep_prob = 1.0})

    loss_t = sess.run([loss], feed_dict)
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
    
    if mux_network:
        S = train_s[batch_id*batch_size:(batch_id+1)*batch_size]
        feed_dict.update({enc_inp[seq_length]: S})

    if args.dropout != 0:
        feed_dict.update({keep_prob = 1.0-args.dropout})

    _, loss_t = sess.run([train_op, loss], feed_dict)
        
    return loss_t


# # Train
# 
# Do not initialize variables if restoring from a saved file.  
# **Warning:** epoch numbers start from 0, and *will* overwrite your old saves!

# In[ ]:

step = resume_at
best_test_repair = 0

if not args.skip_training:
    for t in range(resume_epoch, epochs):
        # Training
        start = time.time()
        train_loss = []
        
        for i in range(resume_training_minibatch, num_train/batch_size):
            f_loss = train_batch(i)
            train_loss.append(f_loss)
            
            # Print progress
            step += 1
            
            print "Step: %d\tEpoch: %g\tLoss: %g" % (step, t + float(i+1)/(num_train/batch_size), train_loss[-1])
            sys.stdout.flush()

            # Checkpoint
            if step % ckpt_every == 0:
                saver.save(sess, os.path.join(ckpt_folder, 'saved-model-attn'), global_step=step)
                print "[Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t, i)
                sys.stdout.flush()
            
        train_loss = np.mean(train_loss)
        resume_training_minibatch = 0

        # Checkpoint before going into validation/testing
        if step % ckpt_every != 0:
            saver.save(sess, os.path.join(ckpt_folder, 'saved-model-attn'), global_step=step)
            print "[Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t+1, 0)
            sys.stdout.flush()
        
        print "End of Epoch: %d" % (t+1)
        print "[Training] Loss: %g" % (train_loss)
        sys.stdout.flush()

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
            
        valid_loss   = np.mean(valid_loss)
        valid_token  = np.mean(valid_token)
        valid_local  = np.mean(valid_local)
        valid_repair = np.mean(valid_repair)
        
        # Print epoch step and validation information
        print "[Validation] Loss: %g Token: %g Localization: %g Repair: %g" % (valid_loss, valid_token, valid_local, valid_repair)
        sys.stdout.flush()

        if test_data_present:
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
                
            test_loss   = np.mean(test_loss)
            test_token  = np.mean(test_token)
            test_local  = np.mean(test_local)
            test_repair = np.mean(test_repair)

            print "[Test] Loss: %g Token: %g Localization: %g Repair: %g" % (test_loss, test_token, test_local, test_repair)
            sys.stdout.flush()

            if test_repair > best_test_repair:
                best_test_repair = test_repair
                copyfile(os.path.join(ckpt_folder, 'saved-model-attn-%d' % step), os.path.join(os.path.join(ckpt_folder, 'best'), 'saved-model-attn-%d' % step))
                print "[Best Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t+1, 0)
                sys.stdout.flush()
        else:
            print "[Test] No test data present"
            sys.stdout.flush()

            if valid_repair > best_test_repair:
                best_test_repair = valid_repair
                copyfile(os.path.join(ckpt_folder, 'saved-model-attn-%d' % step), os.path.join(os.path.join(ckpt_folder, 'best'), 'saved-model-attn-%d' % step))
                print "[Best Checkpoint] Checkpointed at Epoch %d, Minibatch %d." % (t+1, 0)
                sys.stdout.flush()

        print "[Time] Took %g minutes to run." % ((time.time() - start)/60)
        sys.stdout.flush()
else:
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
        
    valid_loss   = np.mean(valid_loss)
    valid_token  = np.mean(valid_token)
    valid_local  = np.mean(valid_local)
    valid_repair = np.mean(valid_repair)
    
    # Print epoch step and validation information
    print "[Validation] Loss: %g Token: %g Localization: %g Repair: %g" % (valid_loss, valid_token, valid_local, valid_repair)
    sys.stdout.flush()

    if test_data_present:
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
            
        test_loss   = np.mean(test_loss)
        test_token  = np.mean(test_token)
        test_local  = np.mean(test_local)
        test_repair = np.mean(test_repair)

        print "[Test] Loss: %g Token: %g Localization: %g Repair: %g" % (test_loss, test_token, test_local, test_repair)
        sys.stdout.flush()
    else:
        print "[Test] No test data present"
        sys.stdout.flush()

# In[ ]:

sess.close()

