import torch
from torch import nn


class mLSTMCell(torch.nn.Module):
    """
    A simple LSTM cell network for educational AI-summer purposes
    
    @article{adaloglou2020rnn,
    title   = "Recurrent neural networks: building a custom LSTM cell",
    author  = "Adaloglou, Nikolas and Karagiannakos, Sergios ",
    journal = "https://theaisummer.com/",
    year    = "2020",
    url     = "https://theaisummer.com/understanding-lstm"
  }
    """

    def __init__(self, input_length=3, hidden_length=20, batch_size=32,
                 device=torch.device('cuda')):
        super(mLSTMCell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()
        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()
        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()
        # out gate components
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()
        self.activation_final = nn.Tanh()
        self.hidden = torch.tensor([[float(0.)] * hidden_length for _ in range(batch_size)],
                                   device=device)
        self.cellState = torch.tensor([[float(0.)] * hidden_length for _ in range(batch_size)],
                                      device=device)

    def initializeHidden(self, h0, c0):
        self.hidden = h0
        self.cellState = c0

    def forget(self, x, h):
        x = self.linear_forget_w1(x)
        h = self.linear_forget_r1(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x, h):
        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)
        return i

    def cell_memory_gate(self, i, f, x, h, c_prev):
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)
        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i
        # forget old context/cell info
        c = f * c_prev
        # learn new context/cell info
        c_next = g + c
        return c_next

    def out_gate(self, x, h):
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)

    def forward(self, x, tuple_in, convergence):
        """
        convergence: the convergence flag, (1 if convergence else 0)
        """
        # if convergence:
        #     (self.hidden, self.cellState) = tuple_in
        # else:
        #     pass
        # h, c_prev = self.hidden, self.cellState

        self.hidden = self.hidden * (1 - convergence) + tuple_in[0] * convergence
        self.cellState = self.cellState * (1 - convergence) + tuple_in[1] * convergence

        # Equation 1. input gate
        i = self.input_gate(x, self.hidden)
        # Equation 2. forget gate
        f = self.forget(x, self.hidden)
        # Equation 3. updating the cell memory
        c_next = self.cell_memory_gate(i, f, x, self.hidden, self.cellState)
        # Equation 4. calculate the main output gate
        o = self.out_gate(x, self.hidden)
        # Equation 5. produce next hidden output
        h_next = o * self.activation_final(c_next)
        return h_next, c_next
