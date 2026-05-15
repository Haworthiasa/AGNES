try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch is not installed. Please install it with: pip install torch")
    import sys
    sys.exit(1)

import numpy as np

# =============================================================================
# Minimal PyTorch implementation of a Seq2Seq model with attention
# following the described architecture:
#   - Bidirectional GRU encoder
#   - Decoder with GRU and attention (Bahdanau-style)
#   - Alignment model: feedforward neural network
#   - Output layer: maxout hidden layer + softmax
# =============================================================================

# -------------------------------
# Device and reproducibility
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# -------------------------------
# Model Components
# -------------------------------

class EncoderBidirGRU(nn.Module):
    """Bidirectional GRU encoder.
       Converts input sequence of indices into annotation vectors
       h_j = [→h_j ; ←h_j] (concatenation of forward and backward states).
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=False       # input shape: (seq_len, batch, ...)
        )
        self.hidden_size = hidden_size  # hidden size per direction

    def forward(self, src):  # src shape: (seq_len, batch)
        # Embed input
        embedded = self.embed(src)  # (seq_len, batch, embed_dim)
        # Run bidirectional GRU
        outputs, hidden = self.gru(embedded)  # outputs: (seq_len, batch, hidden*2)
        # hidden: (num_layers * 2, batch, hidden) – we won't use it here
        return outputs  # annotations (seq_len, batch, hidden*2)


class Attention(nn.Module):
    """Bahdanau-style attention: a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j).
       Computes energy scores and normalizes with softmax.
    """
    def __init__(self, dec_hidden_size, enc_hidden_size):
        super().__init__()
        # Linear transformations for alignment model
        self.W_a = nn.Linear(dec_hidden_size, dec_hidden_size, bias=False)
        self.U_a = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)
        self.v_a = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        """
        Args:
            dec_hidden: (batch, dec_hidden_size) – previous decoder hidden state s_{i-1}
            enc_outputs: (seq_len, batch, enc_hidden_size) – all annotations h_j
        Returns:
            context: (batch, enc_hidden_size) – weighted sum of annotations
            weights: (batch, seq_len) – attention probabilities
        """
        # Expand dec_hidden to match time steps: (batch, dec_hidden) -> (seq_len, batch, dec_hidden)
        dec_hidden_expanded = dec_hidden.unsqueeze(0).expand(enc_outputs.size(0), -1, -1)
        # Compute energy scores
        energy = self.W_a(dec_hidden_expanded) + self.U_a(enc_outputs)   # (seq_len, batch, dec_hidden)
        energy = torch.tanh(energy)
        energy = self.v_a(energy)                                        # (seq_len, batch, 1)
        energy = energy.squeeze(2).transpose(0, 1)                      # (batch, seq_len)
        # Normalize
        weights = torch.softmax(energy, dim=-1)                         # (batch, seq_len)
        # Weighted sum of annotations: (batch, 1, seq_len) @ (batch, seq_len, enc_hidden) -> (batch, enc_hidden)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs.transpose(0, 1)).squeeze(1)
        return context, weights


class MaxOut(nn.Module):
    """Maxout layer: each output unit is the max over k groups of linear outputs.
       Here we use k=2 as in the original paper.
    """
    def __init__(self, in_features, out_features, pool_size=2):
        super().__init__()
        self.out_features = out_features
        self.pool_size = pool_size
        self.linear = nn.Linear(in_features, out_features * pool_size)

    def forward(self, x):
        # x: (batch, in_features)
        output = self.linear(x)   # (batch, out_features * pool_size)
        output = output.view(-1, self.out_features, self.pool_size)
        result, _ = torch.max(output, dim=2)  # max over pool_size
        return result


class DecoderWithAttention(nn.Module):
    """Decoder GRU with attention and maxout output layer.
       Generates one target word at a time.
    """
    def __init__(self, vocab_size, embed_dim, dec_hidden_size, enc_hidden_size, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        # GRU cell: input = concatenation of previous word embedding and context
        self.gru_cell = nn.GRUCell(input_size=embed_dim + enc_hidden_size,
                                   hidden_size=dec_hidden_size)

        # Attention
        self.attention = Attention(dec_hidden_size, enc_hidden_size)

        # Output feedforward: combine y_{i-1}, s_i, c_i -> maxout -> softmax
        # Input size to maxout: embed_dim + dec_hidden_size + enc_hidden_size
        self.maxout = MaxOut(in_features=embed_dim + dec_hidden_size + enc_hidden_size,
                             out_features=dec_hidden_size, pool_size=2)
        self.output_fc = nn.Linear(dec_hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward_step(self, prev_word, prev_hidden, enc_outputs):
        """One decoding step of the decoder.
        Args:
            prev_word: (batch) indices of previous target word (y_{i-1})
            prev_hidden: (batch, dec_hidden_size) – previous decoder state s_{i-1}
            enc_outputs: (seq_len, batch, enc_hidden_size) – annotations
        Returns:
            logits: (batch, vocab_size) – unnormalised probabilities for y_i
            new_hidden: (batch, dec_hidden_size) – s_i
            attention_weights: (batch, seq_len) – alpha_ij
        """
        # Embed previous word
        prev_emb = self.dropout(self.embed(prev_word))  # (batch, embed_dim)

        # Compute attention context based on s_{i-1}
        context, attn_weights = self.attention(prev_hidden, enc_outputs)  # (batch, enc_hidden)

        # Concatenate input to GRU cell
        rnn_input = torch.cat([prev_emb, context], dim=1)  # (batch, embed_dim + enc_hidden)

        # Update decoder state
        new_hidden = self.gru_cell(rnn_input, prev_hidden)  # (batch, dec_hidden)

        # Output layer: combine y_{i-1}, s_i, c_i and apply maxout + softmax
        combined = torch.cat([prev_emb, new_hidden, context], dim=1)  # (batch, embed_dim + dec_hidden + enc_hidden)
        maxout_out = self.maxout(combined)                           # (batch, dec_hidden)
        logits = self.output_fc(maxout_out)                          # (batch, vocab_size)

        return logits, new_hidden, attn_weights

    def forward(self, trg, enc_outputs, teacher_forcing_ratio=1.0):
        """
        Args:
            trg: (target_len, batch) – target sequences including <SOS> and <EOS>
            enc_outputs: (seq_len, batch, enc_hidden) – encoder annotations
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (target_len-1, batch, vocab_size) – logits for each step
        """
        target_len, batch_size = trg.shape
        outputs = torch.zeros(target_len - 1, batch_size, self.output_fc.out_features).to(device)
        # Initial decoder state: zeros or from encoder? We'll use zeros.
        dec_hidden = torch.zeros(batch_size, self.dec_hidden_size).to(device)
        # First input is SOS (index 1, but we assume index 0 is pad, 1 is SOS)
        input_word = trg[0, :]  # (batch)
        for t in range(1, target_len):  # iterate excluding SOS
            logits, dec_hidden, _ = self.forward_step(input_word, dec_hidden, enc_outputs)
            outputs[t-1] = logits
            # teacher forcing or use predicted
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_word = trg[t, :]  # ground truth
            else:
                # greedy: take highest probability
                input_word = logits.argmax(dim=1)  # (batch)
        return outputs


# -------------------------------
# Full Seq2Seq model
# -------------------------------

class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with bidirectional GRU encoder,
       GRU decoder with attention, and maxout output layer.
    """
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.encoder = EncoderBidirGRU(src_vocab_size, embed_dim, hidden_size)
        # Encoder outputs have dimension hidden*2 (bidirectional)
        enc_out_dim = hidden_size * 2
        self.decoder = DecoderWithAttention(
            vocab_size=trg_vocab_size,
            embed_dim=embed_dim,
            dec_hidden_size=hidden_size,
            enc_hidden_size=enc_out_dim
        )

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # Encoder forward
        enc_outputs = self.encoder(src)  # (src_len, batch, enc_out_dim)
        # Decoder forward
        output = self.decoder(trg, enc_outputs, teacher_forcing_ratio)
        return output


# -------------------------------
# Data preparation – synthetic
# -------------------------------

def generate_synthetic_data(num_samples, src_vocab, trg_vocab, max_len=5):
    """Create random source and target sequences of varying length."""
    src_data = []
    trg_data = []
    for _ in range(num_samples):
        src_len = np.random.randint(2, max_len+1)
        trg_len = np.random.randint(2, max_len+1)
        src_seq = [np.random.randint(2, src_vocab) for _ in range(src_len)]
        trg_seq = [np.random.randint(2, trg_vocab) for _ in range(trg_len)]
        src_data.append(src_seq)
        trg_data.append(trg_seq)
    return src_data, trg_data


def collate_pad(batch, src_vocab_size, trg_vocab_size):
    """Pad sequences to the same length and add SOS/EOS.
       Uses 0 for padding, 1 for SOS, 2 for EOS.
    """
    src_sents = [pair[0] for pair in batch]   # FIX: extract source sentences properly
    trg_sents = [pair[1] for pair in batch]   # and target sentences
    # Add SOS and EOS to target: start with 1, end with 2, also for source?
    src_seqs = [[1] + sent + [2] for sent in src_sents]
    trg_seqs = [[1] + sent + [2] for sent in trg_sents]
    # Pad to max length
    src_len = max(len(s) for s in src_seqs)
    trg_len = max(len(s) for s in trg_seqs)
    src_padded = torch.zeros((len(src_seqs), src_len), dtype=torch.long)
    trg_padded = torch.zeros((len(trg_seqs), trg_len), dtype=torch.long)
    for i, s in enumerate(src_seqs):
        src_padded[i, :len(s)] = torch.tensor(s)
    for i, t in enumerate(trg_seqs):
        trg_padded[i, :len(t)] = torch.tensor(t)
    # Transpose to (seq_len, batch) as expected by model
    return src_padded.T, trg_padded.T


# -------------------------------
# Training and evaluation demo
# -------------------------------

def train_epoch(model, src_padded, trg_padded, optimizer, criterion, clip=1.0):
    model.train()
    optimizer.zero_grad()
    # teacher forcing ratio decays? We'll keep 0.5
    output = model(src_padded, trg_padded, teacher_forcing_ratio=0.5)
    # output shape: (trg_len-1, batch, trg_vocab)
    loss = criterion(output.reshape(-1, output.shape[-1]),
                     trg_padded[1:].reshape(-1))  # ignore SOS
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.item()


def evaluate(model, src_padded, trg_padded, criterion):
    model.eval()
    with torch.no_grad():
        output = model(src_padded, trg_padded, teacher_forcing_ratio=0)  # no teacher forcing
        loss = criterion(output.reshape(-1, output.shape[-1]),
                         trg_padded[1:].reshape(-1))
    return loss.item()


def main():
    # Hyperparameters
    SRC_VOCAB = 10
    TRG_VOCAB = 10
    EMBED_DIM = 16
    HIDDEN_SIZE = 32
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 0.001

    # Generate synthetic data
    src_data, trg_data = generate_synthetic_data(100, SRC_VOCAB, TRG_VOCAB)
    # Create batches
    dataset = list(zip(src_data, trg_data))
    batches = [dataset[i:i+BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]

    # Model, criterion, optimizer
    model = Seq2Seq(SRC_VOCAB, TRG_VOCAB, EMBED_DIM, HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    print("Starting training...\n")

    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        for batch in batches:
            src_padded, trg_padded = collate_pad(batch, SRC_VOCAB, TRG_VOCAB)
            src_padded = src_padded.to(device)
            trg_padded = trg_padded.to(device)
            loss = train_epoch(model, src_padded, trg_padded, optimizer, criterion)
            epoch_loss += loss
        avg_loss = epoch_loss / len(batches)
        # Evaluate on first batch without teacher forcing
        src_padded, trg_padded = collate_pad(batches[0], SRC_VOCAB, TRG_VOCAB)
        src_padded = src_padded.to(device)
        trg_padded = trg_padded.to(device)
        eval_loss = evaluate(model, src_padded, trg_padded, criterion)
        print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.3f} | Eval Loss: {eval_loss:.3f}")

    print("\nTraining finished.\n")

    # Quick inference demo: greedy decoding
    model.eval()
    src_sample, trg_sample = dataset[0]
    src_tensor = torch.tensor([[1] + src_sample + [2]]).T.to(device)  # (seq_len, 1)
    with torch.no_grad():
        enc_outputs = model.encoder(src_tensor)  # (src_len, 1, enc_hidden)
        # Initialise decoder
        dec_hidden = torch.zeros(1, model.decoder.dec_hidden_size).to(device)
        input_word = torch.tensor([1]).to(device)   # SOS
        predicted_ids = []
        for _ in range(12):  # max length arbitrary
            logits, dec_hidden, attn = model.decoder.forward_step(input_word, dec_hidden, enc_outputs)
            predicted = logits.argmax(dim=1).item()
            if predicted == 2:  # EOS
                break
            predicted_ids.append(predicted)
            input_word = torch.tensor([predicted]).to(device)
    print("Example translation:")
    print(f"Source   : {src_sample}")
    print(f"Predicted: {predicted_ids}")
    print(f"Target   : {trg_sample}")

if __name__ == "__main__":
    main()