# Machine-translation-English--French-with-Deep-neural-Network
 Build a deep neural network that functions as part of an end-to-end machine translation pipeline


The pipeline will accept English text as input and return the French translation.

Steps:
- Preprocess - Convert text to sequence of integers.
- Models - Create models which accepts a sequence of integers as input and returns a probability distribution over possible translations
- Prediction - Run the model on English text to return French translation.

# Introduction

Machine Translation can be thought of as a sequence-to-sequence learning problem.
You have one sequence going in, i.e. a sentence in the source language, and one sequence coming out, its translation in the target language.
This seems like a very hard problem. But recent advances in Recurrent Neural Networks have shown a lot of improvement. A typical approach is to use a recurrent layer to encode the meaning of the sentence by processing the words in a sequence, and then either use a dense or fully-connected layer to produce the output, or use another decoding layer.
Experimenting with different network architectures and recurrent layer units (such as LSTMs, GRUs, etc.), you can come up with a fairly simple model that performs decently well on small-to-medium size datasets. Commercial-grade translation systems need to deal with a much larger vocabulary, and hence have to use a much more complex model, apply different optimizations, etc. Training such models requires a lot of data and compute time.

For a neural network to predict on text data, it first has to be turned into data it can understand. Text data like "dog" is a sequence of ASCII character encodings. Since a neural network is a series of multiplication and addition operations, the input data needs to be number(s).

We can turn each character into a number or each word into a number. These are called character and word ids, respectively. Character ids are used for character level models that generate text predictions for each character. For a simple characterer level model, you can visit my other project about [decyphering a code with a simple RNN](). A word level model uses word ids that generate text predictions for each word. Word level models tend to learn better, since they are lower in complexity
