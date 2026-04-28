import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(input_size=embed_size, hidden_size = hidden_size, num_layers = rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, input length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        embeddings = self.embedding(indices)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first = True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embeds)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first = True)
        logits = self.linear(outputs)
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        device = next(self.parameters()).device

        bos = [self.dataset.bos_id]
        pref_coded = bos + self.dataset.text2ids(prefix)
        pref_tensor = torch.tensor([pref_coded], dtype=torch.long, device=device)
        embeds_pref = self.embedding(pref_tensor)

        already_tokens = list(pref_coded)

        output, hidden = self.rnn(embeds_pref)
        logits = self.linear(output)
        last_logits = logits[0, -1, :]

        probs = torch.softmax(last_logits/temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        already_tokens.append(next_token)

        while next_token != self.dataset.eos_id and len(already_tokens) < self.max_length:
            input_next = torch.tensor([[next_token]], dtype=torch.long, device=device)
            next_embed = self.embedding(input_next)
            output, hidden = self.rnn(next_embed, hidden)
            logits = self.linear(output)

            last_logits = logits[0, -1, :]

            probs = torch.softmax(last_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            already_tokens.append(next_token)

        generated = self.dataset.ids2text(already_tokens)
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        return generated
