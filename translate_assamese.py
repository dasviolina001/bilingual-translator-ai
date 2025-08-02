import torch
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer and vocab utilities
def tokenize(sentence):
    return sentence.strip().lower().split()

def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode(sentence, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in tokenize(sentence)]

def decode(indices, inv_vocab):
    words = []
    for idx in indices:
        word = inv_vocab.get(idx, '<UNK>')
        if word == '<EOS>':
            break
        if word not in ['<SOS>', '<PAD>']:
            words.append(word)
    return ' '.join(words)

# Model definitions
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        hidden = hidden.view(1, -1, hidden.size(2) * 2)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = attention
        self.rnn = nn.GRU(hidden_dim * 2 + embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim + embed_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden

# Load model and vocab
df = pd.read_csv("asmm_engg_cleaned.csv").fillna("")
eng_vocab = build_vocab(df['eng'])
asm_vocab = build_vocab(df['asm'])
inv_asm_vocab = {v: k for k, v in asm_vocab.items()}

embed_size = 256
hidden_size = 512
encoder = Encoder(len(eng_vocab), embed_size, hidden_size).to(device)
attention = Attention(hidden_size).to(device)
decoder = Decoder(len(asm_vocab), embed_size, hidden_size, attention).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

encoder.eval()
decoder.eval()

# 🔁 Exported translate function
def translate(text):
    tokens = encode(text, eng_vocab)
    src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = encoder(src_tensor)
        hidden = hidden[:, :, :hidden_size] + hidden[:, :, hidden_size:]
        input_token = torch.tensor([asm_vocab['<SOS>']], dtype=torch.long).to(device)

        translated_indices = []
        for _ in range(50):
            output, hidden = decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)
            translated_indices.append(top1.item())
            input_token = top1
            if top1.item() == asm_vocab['<EOS>']:
                break

    return decode(translated_indices, inv_asm_vocab)
