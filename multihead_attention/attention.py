import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wout = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, attention_mask=None):
        q = self.split_heads(self.Wq(hidden_states))
        k = self.split_heads(self.Wk(hidden_states))
        v = self.split_heads(self.Wv(hidden_states))

        attention = self.scaled_dot_attention(q, k, v, attention_mask)
        attention = self.concatenate_heads(attention)

        out = self.Wout(attention)

        return out

    def concatenate_heads(self, attention_states: torch.Tensor):
        batch_size, _, sequence_len, _ = attention_states.size()
        attention_states = (
            attention_states.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_len, self.d_model)
        )

        # print("Attention states: ", attention_states)

        return attention_states

    def scaled_dot_attention(self, q, k, v, attention_mask: torch.Tensor = None):
        k = k.transpose(2, 3)  # [batch_size, num_heads, d_k, seq_len]
        # print("Q: ", q)
        # print("K: ", k)
        out = q @ k
        out = out / torch.sqrt(
            torch.tensor(self.d_k)
        )  # [batch_size, num_heads, seq_len, seq_len]
        if attention_mask is not None:
            out = out.masked_fill(attention_mask == 0, float("-inf"))
        out = torch.softmax(out, dim=-1) @ v
        return out

    def split_heads(self, hidden_states: torch.Tensor):
        batch_size, num_sequences, _ = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, num_sequences, self.num_heads, self.d_k
        )

        hidden_states = hidden_states.transpose(1, 2)
        # WHY DO WE TRANSPOSE?
        # we transpose to calculate attention between head in every element in sequence
        # if we do not transpose we make calculations only inside one sequence
        return hidden_states


if __name__ == "__main__":
    attention = Attention(8, 2)
    hidden_states = torch.rand(1, 5, 8)
    attention_mask = torch.ones(2, 5, 5)
    attention_mask[:, :, 3:] = 0
    attention_hidden_states = attention(hidden_states, attention_mask)
