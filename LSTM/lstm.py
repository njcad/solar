"""
lstm.py

Description: main Long Short Term Memory (LSTM) learning model for solar power prediction given weather data. 
For Stanford CS221.

Date: 22 November 2023

Authors: Nate Cadicamo, Jessica Yang, Riya Karumanchi, Ira Thawornbut

Approach: 


"""

# libraries to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import get_data as GD 


# PART 1: load, preprocess, normalize data

# define data file paths
power1 = "../data/Plant_1_Generation_Data.csv"
weather1 = "../data/Plant_1_Weather_Sensor_Data.csv"
power2 = "../data/Plant_2_Generation_Data.csv"
weather2 = "../data/Plant_2_Weather_Sensor_Data.csv"

# get sorted, normalized, split sequential data from get_data.py
X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = GD.sort_data(weather1, power1)
X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = GD.sort_data(weather2, power2)




# PART 2: build LSTM components
class LSTMcell:
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the LSTM cell parameters.

        :param input_dim: Dimension of the input vector.
        :param hidden_dim: Dimension of the hidden state.
        """
        self.input_dim = input_dim #define what the input dimension looks like
        self.hidden_dim = hidden_dim #define hidden dimension
        # for reference, tf default was 50, we can start with that 

        # Forget gate parameters
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.1
        self.bf = np.zeros((hidden_dim, 1))

        # Input gate parameters
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.1
        self.bi = np.zeros((hidden_dim, 1))

        # Cell state parameters
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.1
        self.bc = np.zeros((hidden_dim, 1))

        # Output gate parameters
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.1
        self.bo = np.zeros((hidden_dim, 1))

        # Output parameters
        self.Wy = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.by = np.zeros((hidden_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, xt, a_prev, c_prev):
        """
        Forward pass for a single LSTM cell.

        :param xt: Input vector at the current time step.
        :param a_prev: Hidden state vector from the previous time step.
        :param c_prev: Cell state vector from the previous time step.
        :return: Next hidden state, next cell state, and the current output.
        """
        # Concatenate a_prev and xt
        concat = np.vstack((a_prev, xt))

        # Forget gate: decides what information to discard from the cell state
        ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # Input gate: decides what new information to store in the cell state
        it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)

        # Cell gate: creates a vector of new candidate values to add to the state
        cct = self.tanh(np.dot(self.Wc, concat) + self.bc)

        # Update the cell state
        c_next = ft * c_prev + it * cct

        # Output gate: decides what part of the cell state to output
        ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

        # Update the hidden state
        a_next = ot * self.tanh(c_next)

        # Compute the output of the cell
        yt = np.dot(self.Wy, a_next) + self.by

        # Return next hidden state, cell state, output, and gate activations
        return a_next, c_next, yt, (ft, it, cct, ot)

    # Forward pass through sequence
    def forward_pass_through_sequence(LSTM, X):
        """
        Forward pass through an entire sequence of data using the LSTM cell.

        :param LSTM: An instance of the LSTMcell class.
        :param X: Input data sequence (sequence_length x input_dim).
        :return: Array of outputs for each time step in the sequence.
        """
        outputs = []
        a_prev = np.zeros((LSTM.hidden_dim, 1))  # List storing all hidden states
        c_prev = np.zeros((LSTM.hidden_dim, 1))  # List storing all cell states
        gate_activations = []                    # Store gate activations

        for xt in X:
            a_prev, c_prev, yt = LSTM.forward(xt, a_prev, c_prev)
            outputs.append(yt)
            a_states.append(a_prev)
            c_states.append(c_prev)
            gate_activations.append(gates) # Append intermediate gate activations to respective lists

        return np.array(outputs), a_states, c_states, gate_activations

    # Backward Pass for simple LSTM cell.
    # dy: gradient of loss w.r.t. output of the current cell
    # da_next: gradient of loss w.r.t. to the next hidden state
    # dc_next: gradient of loss w.r.t. the next cell state
    # xt: input data at current time step
    # a_prev: previous hidden state 
    # c_prev: previous cell state
    # returns gradients of loss w.r.t to above parameters and states
    def lstm_cell_backward(self, dy, da_next, dc_next, xt, a_prev, c_prev, ft, it, cct, ot, c_next):
        # Concatenate a_prev and xt
        concat = np.vstack((a_prev, xt))

        # Gradients of the loss with respect to the gates activations
        dot = dy * self.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * dy * (1 - self.tanh(c_next) ** 2)) * (1 - cct ** 2)
        dit = (dc_next * cct + ot * dy * (1 - self.tanh(c_next) ** 2)) * it * (1 - it)
        dft = (dc_next * c_prev + ot * dy * (1 - self.tanh(c_next) ** 2)) * ft * (1 - ft)

        # Compute gradients w.r.t parameters of each gate
        dWf = np.dot(dft, concat.T)
        dWi = np.dot(dit, concat.T)
        dWc = np.dot(dcct, concat.T)
        dWo = np.dot(dot, concat.T)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute gradients w.r.t previous hidden state and cell state
        da_prev = np.dot(self.Wf[:, :self.hidden_dim].T, dft) + \
                np.dot(self.Wi[:, :self.hidden_dim].T, dit) + \
                np.dot(self.Wc[:, :self.hidden_dim].T, dcct) + \
                np.dot(self.Wo[:, :self.hidden_dim].T, dot) + \
                da_next
        dc_prev = ft * dc_next + \
                it * cct * dc_next + \
                ot * dy * (1 - self.tanh(c_next) ** 2) + \
                self.sigmoid(np.dot(self.Wf, concat) + self.bf) * c_prev * dc_next

        # Return gradients
        return da_prev, dc_prev, dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo

    # Backpropogation Through Time (BPTT) 
    def bptt(self, X, Y):
        # Initialize gradients as zero
        dWf, dWi, dWc, dWo, dWy = [np.zeros_like(W) for W in (self.Wf, self.Wi, self.Wc, self.Wo, self.Wy)]
        dbf, dbi, dbc, dbo, dby = [np.zeros_like(b) for b in (self.bf, self.bi, self.bc, self.bo, self.by)]

        # Initialize gradients w.r.t. next hidden and cell states
        da_next = np.zeros((self.hidden_dim, 1))
        dc_next = np.zeros((self.hidden_dim, 1))

        # Forward pass through the sequence to get all hidden states, cell states, and gate activations
        outputs, a_states, c_states, gate_activations = forward_pass_through_sequence(self, X)

        # Process the sequence backwards
        for t in reversed(range(len(X))):
            # Compute gradients w.r.t output at time step t
            dy = (outputs[t] - Y[t]) 

            # Extract the gate activations for this time step
            ft, it, cct, ot = gate_activations[t]
            a_prev = a_states[t-1] if t != 0 else np.zeros((self.hidden_dim, 1))
            c_prev = c_states[t-1] if t != 0 else np.zeros((self.hidden_dim, 1))

            # Backpropagate through LSTM cell
            da_prev, dc_prev, dWf_t, dWi_t, dWc_t, dWo_t, dbf_t, dbi_t, dbc_t, dbo_t = \
                self.lstm_cell_backward(dy, da_next, dc_next, X[t], a_prev, c_prev, ft, it, cct, ot, c_states[t])
            
            # Update gradients w.r.t the parameters
            dWf += dWf_t
            dWi += dWi_t
            dWc += dWc_t
            dWo += dWo_t

            dbf += dbf_t
            dbi += dbi_t
            dbc += dbc_t
            dbo += dbo_t

            # Update da_next and dc_next for next iteration of the loop
            da_next = da_prev
            dc_next = dc_prev

        # Update LSTM parameters using the gradients
        self.Wf -= learning_rate * dWf
        self.Wi -= learning_rate * dWi
        self.Wc -= learning_rate * dWc
        self.Wo -= learning_rate * dWo
        self.Wy -= learning_rate * dWy

        self.bf -= learning_rate * dbf
        self.bi -= learning_rate * dbi
        self.bc -= learning_rate * dbc
        self.bo -= learning_rate * dbo
        self.by -= learning_rate * dby


# Train model
def train_model(LSTM, data, epochs, learning_rate):
    """
    Train the LSTM model.

    :param LSTM: An instance of the LSTMcell class.
    :param data: Training data.
    :param epochs: Number of epochs to train for.
    :param learning_rate: Learning rate for updates.
    """
    for epoch in range(epochs):
        total_loss = 0

        for X, y_true in data:
            # Forward pass
            y_pred = forward_pass_through_sequence(LSTM, X)

            # Compute loss (assuming MSE)
            loss = np.mean((y_pred - y_true) ** 2) # this uses mean squared loss but we could also do our own loss function
            # loss = compute_loss(y_pred, y_true)

            total_loss += loss

            # Backpropagation through time (BPTT)
            # TODO: Implement BPTT to update LSTM parameters
            bptt(X, y_true)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")
