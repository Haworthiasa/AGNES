# Neural Machine Translation by Jointly Learning to Align and Translate

## URL
https://arxiv.org/abs/1409.0473

## One-Sentence
Not available

## Architecture
{"encoder": "Bidirectional RNN (biRNN) with forward and backward recurrent neural networks (GRU-based). Each word x_j is encoded into an annotation h_j = [\u2192h_j; \u2190h_j] concatenating the forward and backward hidden states, capturing context from both directions.", "decoder": "Recurrent neural network (GRU) with attention. At each step i, it computes a context vector c_i = \u03a3_j \u03b1_{ij} h_j, where \u03b1_{ij} is a soft weight from an alignment model. The probability of target word y_i is conditional on the previous word y_{i-1}, decoder state s_i, and context c_i.", "alignment_model": "Feedforward neural network a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j). The outputs e_{ij} are normalized via softmax to produce attention weights \u03b1_{ij}.", "output_layer": "A maxout hidden layer followed by a softmax to compute the probability of the next target word."}

## Results
Not available

## Questions
### COMPREHENSION
Explain how the encoder in this architecture generates annotations \( h_j \) for each input word \( x_j \). Specifically, describe how the forward and backward hidden states are combined and why this bidirectional approach is beneficial for capturing context from both directions. Then, trace how the decoder uses these annotations to compute the context vector \( c_i \) at each step \( i \).

### ANALYSIS
Consider a what-if scenario where the alignment model is changed from a feedforward network (\( a(s_{i-1}, h_j) = v_a^T \tanh(W_a s_{i-1} + U_a h_j) \)) to a simpler dot-product scoring (\( e_{ij} = s_{i-1}^T h_j \)). How might this alteration impact the model’s ability to learn alignment weights, especially in tasks with long input sequences or complex dependencies? Discuss potential advantages in computational efficiency versus possible losses in expressiveness.

### SYNTHESIS
Connect this attention mechanism to the self-attention used in Transformer models. Compare the roles of the alignment model here (which aligns decoder states with encoder annotations) with the scaled dot-product attention in Transformers (which computes attention between all positions in a sequence). How do these different approaches handle sequential dependencies, and what trade-offs exist in terms of parallelization, memory usage, and ability to capture long-range relationships?

### CHALLENGE
Implement a simplified version of the output layer: given a context vector \( c_i \) and decoder state \( s_i \), compute the probability distribution over target words using a maxout hidden layer followed by a softmax. Provide pseudocode for the forward pass, including the maxout operation (e.g., taking the maximum over groups of hidden units). Then, discuss potential implementation challenges, such as efficiently computing the softmax over a large vocabulary or avoiding gradient issues with the maxout activation.

## Demo Status
failed