import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import math

# Read train text
with open('train.txt') as f:
    trainText = f.read()

# Obtain words in the train text
trainText = trainText.replace('\n', ' ').split(' ')
vocabProcessor = learn.preprocessing.VocabularyProcessor(1)
trainWords = [x[0] for x in vocabProcessor.fit_transform(trainText)]
XTrain, yTrain = trainWords[:-1], trainWords[1:]

# Session variable
sess = tf.InteractiveSession()

# Initialize constants
BATCH = 20
EMBED = 30
H = 100
VOCAB = len(vocabProcessor.vocabulary_)

X = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None])
E = tf.Variable(tf.random_uniform([VOCAB, EMBED], -1, 1))
embd = tf.nn.embedding_lookup(E, X)

# Layer One
w1 = tf.Variable(tf.random_uniform([EMBED, H], minval=-1, maxval=1))
b1 = tf.Variable(tf.constant(0.1, shape=[H]))
h1 = tf.nn.relu(tf.matmul(embd, w1) + b1)

# Layer Two
w2 = tf.Variable(tf.random_uniform([H, VOCAB], minval=-1, maxval=1))
b2 = tf.Variable(tf.constant(0.1, shape=[VOCAB]))

# Logits
logits = tf.matmul(h1, w2) + b2

# Error
error = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)

# Train step using adam optimizer
trainStep = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
loss = tf.reduce_mean(error)

# Run the session
sess.run(tf.initialize_all_variables())

for e in range(len(XTrain) // BATCH - 1):
    trainStep.run(feed_dict={
            X: XTrain[BATCH*e: BATCH *(e + 1)],
            y: yTrain[BATCH*e: BATCH *(e + 1)]
        })

# Read test text
with open('test.txt') as f:
    testText = f.readlines()

# Obtain words in test text
testText = testText.replace('\n', ' ').split(' ')

testWords = [x[0] for x in vocabProcessor.fit_transform(testText)]
XTest, yTest = testWords[:-1], testWords[1:]

print "Perplexity ",math.e ** loss.eval(feed_dict={X: XTest, y: yTest})
