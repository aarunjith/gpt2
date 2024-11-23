import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random

# Load data
with open('names.txt', 'r') as f:
    words = f.read().splitlines()

# Create character mappings
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Constants
CONTEXT_LENGTH = 3
EMBEDDING_DIM = 2
HIDDEN_DIM = 100
BATCH_SIZE = 32

def get_data(words, debug=False):
    X = []
    Y = []
    for word in words:
        context = [stoi['.']] * CONTEXT_LENGTH
        for char in word + '.':
            if debug:
                print(f'{''.join([itos[ix] for ix in context])} --> {char}')
            X.append(context)
            Y.append(stoi[char])
            context = context + [stoi[char]]
            context = context[-CONTEXT_LENGTH:]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(stoi), EMBEDDING_DIM)
        self.layers = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * CONTEXT_LENGTH, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Tanh(),  # Tanh 1
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Tanh(),  # Tanh 2
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Tanh(),  # Tanh 3
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Tanh(),  # Tanh 4
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Tanh(),  # Tanh 5
            nn.Linear(HIDDEN_DIM, len(stoi)),
        )
        
        # Store activations and gradients with unique keys
        self.activations = {}
        self.gradients = {}
        self.tanh_count = 0
        
        # Register hooks for all layers
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Tanh):
                self.tanh_count += 1
                layer_name = f'tanh_{self.tanh_count}'
                layer.register_forward_hook(self._get_activation(layer_name))
                layer.register_full_backward_hook(self._get_gradient(layer_name))
    
    def _get_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _get_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, EMBEDDING_DIM * CONTEXT_LENGTH)
        x = self.layers(x)
        return x

def plot_weight_gradients(layers):
    plt.subplot(2, 2, 1)
    legends = []
    for ix, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            t = layer.weight.grad
            grad_data = (t/layer.weight).mean().item()
            print(f'Weight {tuple(t.shape)}: mean %+.5f, std %.5f, grad:data ratio: %.2f' 
                  % (t.mean(), t.std(), grad_data))
            hy, hx = torch.histogram(t, density=True)
            legends.append(f'Linear - {ix}')
            plt.plot(hx[:-1].detach(), hy.detach())
    plt.title('Weight Gradients Distribution')
    plt.legend(legends)
    plt.grid(True)

def plot_weight_update_ratios(weight_update_ratios):
    plt.subplot(2, 2, 2)
    legends = []

    for layer_ix, ratios in weight_update_ratios.items():
        plt.plot(ratios)
        legends.append(f'Layer {layer_ix} updates/weights ratio')

    plt.xlabel('Training steps')
    plt.ylabel('Update/Weight ratio')
    plt.title('Weight Update Ratios Over Training')
    plt.legend(legends)
    plt.grid(True)

def plot_activation_gradients(layers, collected_gradients):
    plt.subplot(2, 2, 3)
    legends = []
    ix = 1
    
    # Convert collected_gradients to list if it's not already
    gradients_list = list(collected_gradients)
    
    # Count total Tanh layers
    tanh_count = sum(1 for layer in layers if isinstance(layer, nn.Tanh))
    print(f"Total Tanh layers: {tanh_count}")
    
    for layer, grads in zip(layers, gradients_list):
        if isinstance(layer, nn.Tanh):
            t = grads
            print(f'Tanh {ix}: mean %+.5f, std %.5f' % (t.mean(), t.std()))
            hy, hx = torch.histogram(t.flatten(), density=True)
            legends.append(f'Tanh - {ix}')
            plt.plot(hx[:-1].detach(), hy.detach())
            ix += 1
    
    plt.xlabel('Gradient value')
    plt.ylabel('Density')
    plt.title('Distribution of Gradients through Tanh')
    plt.legend(legends)
    plt.grid(True)

def plot_tanh_activations(layers, activations):
    plt.subplot(2, 2, 4)
    legends = []
    ix = 1
    
    # Convert activations to list if it's not already
    activations_list = list(activations)
    
    # Count total Tanh layers
    tanh_count = sum(1 for layer in layers if isinstance(layer, nn.Tanh))
    print(f"Total Tanh layers: {tanh_count}")
    
    for layer, acts in zip(layers, activations_list):
        if isinstance(layer, nn.Tanh):
            t = acts
            print(f'Tanh {ix} activations: mean %+.5f, std %.5f' % (t.mean(), t.std()))
            hy, hx = torch.histogram(t.flatten(), density=True)
            legends.append(f'Tanh - {ix}')
            plt.plot(hx[:-1].detach(), hy.detach())
            ix += 1
    
    plt.xlabel('Activation value')
    plt.ylabel('Density')
    plt.title('Distribution of Tanh Activations')
    plt.legend(legends)
    plt.grid(True)

if __name__ == "__main__":
    model = Network()
    
    # Get training data
    X, Y = get_data(words)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters())
    
    # Track weight updates over time
    weight_update_ratios = {ix: [] for ix, layer in enumerate(model.layers) 
                           if isinstance(layer, nn.Linear)}
    
    # Training loop
    for step in tqdm(range(1000)):  # Adjust number of steps as needed
        # Get random batch
        ix = torch.randint(0, X.shape[0], (BATCH_SIZE,))
        Xbatch, Ybatch = X[ix], Y[ix]
        
        # Forward pass
        logits = model(Xbatch)
        loss = F.cross_entropy(logits, Ybatch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Track weight updates
        for ix, layer in enumerate(model.layers):
            if isinstance(layer, nn.Linear):
                update_ratio = (optimizer.param_groups[0]['lr'] * layer.weight.grad.std() / layer.weight.std()).mean().item()
                weight_update_ratios[ix].append(update_ratio)
        
        optimizer.step()
    
    # Create a single figure for all plots
    plt.figure(figsize=(20, 16))
    
    # Plot all results
    plot_weight_gradients(model.layers)
    plot_weight_update_ratios(weight_update_ratios)
    plot_activation_gradients(model.layers, model.gradients.values())
    plot_tanh_activations(model.layers, model.activations.values())
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
