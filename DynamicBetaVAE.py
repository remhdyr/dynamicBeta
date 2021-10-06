# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:42:51 2020

@author: Klas Rydhmer
"""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.offsetbox as ob

import pickle
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F

# %% Prepare paths and environment
master_path = os.getcwd() + '/'
path_to_data = master_path + 'Data/'
folder_name = 'Test2/'

# If true, beta is not adjusted dynamically
vanilla = False

# If True, the semi-supervision is added
semi_supervised = False

# Plot resolution
dpi = 75

# %% Set up VAE
batch_size = 256
log_interval = 10
device = torch.device("cuda")

# Hyper parameters
a = 0.2
b = 0.05
w1 = 1.2
w2 = 1.2
w3 = 0.9
w4 = 1.1

max_beta = 2

# %% Load data
fft_lists = []
species = []
# Loop over all folders
for file in os.listdir(path_to_data):
    
    # Load FFT pickle into list
    with open(path_to_data + file, 'rb') as handle:
        fft_list = pickle.load(handle)
    
    fft_lists.append(fft_list)
    species.append(file)

# Frequency axis
ftx = np.linspace(0, 2000, 193)


# %% Assemble datasets
# Start with field data
fft_list = fft_lists[np.where(np.array(species) == 'Unlabelled')[0][0]]

# Labelled data
labelled_idxs = np.where(np.array(species) != 'Unlabelled')[0]

# Some are used for semi-supervision
ss_ind = np.random.choice(labelled_idxs, len(labelled_idxs) - 4, replace=False)

# Some are kept for testing
val_ind = labelled_idxs[np.isin(labelled_idxs, ss_ind) == False]

# Assemble semi-supervised set
ss_targets = []
ss_fft = []
for idx in ss_ind:
    fft_ss_list = fft_lists[idx]
    for j in range(500):
        ss_fft.append(fft_ss_list[j])
        ss_targets.append(idx)

        # Also put into the usual traing set
        fft_list.append(fft_ss_list[j])

# What species are included?
ss_species = np.array(species)[ss_ind]

# Assemble semi-supervised validation set
ss_val_targets = []
ss_val_fft = []
for idx in val_ind:
    fft_ss_val_list = fft_lists[idx]
    for j in range(500):
        ss_val_fft.append(fft_ss_val_list[j])
        ss_val_targets.append(idx)

        # Also put into the usual traing set(?)
        fft_list.append(fft_ss_list[j])

# What species are used for testing?
ss_val_species = np.array(species)[val_ind]
print('Validation species:', ss_val_species)

# Convert fft list to numpy array
X = fft_list.copy()
np.random.shuffle(X)
X = np.array(X)

# Empty temporary variables
fft_list = []
time_sigs = []


# %% Define data loader and split into training and test set
def normalize(array, mode='max', multiple=False):
    assert mode in ['max', 'median']

    if multiple is False:
        if mode == 'median':
            return (array - np.min(array))/(np.median(array) - np.min(array))

        if mode == 'max':
            return (array - np.min(array))/(np.max(array) - np.min(array))

    else:
        if mode == 'median':
            top = array - np.min(array, axis=0)
            bottom = np.median(array, axis=0) - np.min(array, axis=0)
            return top/bottom

        if mode == 'max':
            top = array - np.min(array, axis=0)
            bottom = np.max(array, axis=0) - np.min(array, axis=0)
            return top/bottom
    

class insectSpectra(Dataset):
    def __init__(self, data, targets, batch_norm=False, event_norm=True):

        # Initialize
        super().__init__()

        # Normalize each event between 0 and 1
        data = np.log10(data)

        # Normalize over the whole batch
        if batch_norm is True:
            # Otherwise it makes no sense..
            assert event_norm is False
            data = normalize(data, mode='median', multiple=True)

        if event_norm is True:
            for i in range(len(data)):
                event = data[i, :, :]
                event_normed = normalize(event)
                data[i, :, :] = event_normed

        # Convert numpy arrays to tensors
        self.data = torch.FloatTensor(data)

        # Make a target
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        # Method to return number of data points
        return len(self.data)

    def __getitem__(self, index):
        # Method to fetch indexed element
        return self.data[index], self.targets[index].type(torch.LongTensor)


# Split training and test set
N = len(X)
nVal = 3000
nTest = 1000
nTrain = N - (nVal + nTest)

# Shuffle again for good measure
np.random.shuffle(X)

# Create unlabelled datasets
testSet = insectSpectra(data=X[:nTest, :], targets=np.ones(nTest))
valSet = insectSpectra(data=X[nTest:nTest+nVal, :], targets=np.ones(nVal))
trainSet = insectSpectra(data=X[nTest+nVal:, :], targets=np.ones(nTrain))

# Labelled datasets
ss_set = insectSpectra(data=np.array(ss_fft), targets=ss_targets)
ss_val_set = insectSpectra(data=np.array(ss_val_fft), targets=ss_val_targets)

# %% Create dataloaders
train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valSet, batch_size=batch_size, shuffle=True)
ss_loader = DataLoader(ss_set, batch_size=batch_size, shuffle=True)
ss_val_loader = DataLoader(ss_val_set, batch_size=batch_size, shuffle=True)

# Store data shape for plotting etc.
data_shape = train_loader.dataset.data.shape[1:]

# %% Define network
n1 = np.product(data_shape)
n2 = 128
n3 = 128
n4 = 64
n5 = 32
n6 = 32
n7 = 16
n8 = 8
n9 = 4

bottleneck = 2

# VAE object
class VAE(nn.Module):
    def __init__(self, AE=False, beta=1):
        super(VAE, self).__init__()

        self.AE = AE
        self.beta = beta

        # Encoder part
        self.e1 = nn.Linear(n1, n2)
        self.e2 = nn.Linear(n2, n3)
        self.e3 = nn.Linear(n3, n4)
        self.e4 = nn.Linear(n4, n5)
        self.e5 = nn.Linear(n5, n7)
        self.e6 = nn.Linear(n7, n8)
        self.e7 = nn.Linear(n8, n9)

        # Bottleneck
        self.e8_1 = nn.Linear(n9, bottleneck)  # mu
        self.e8_2 = nn.Linear(n9, bottleneck)  # logvar

        # Decoder part
        self.d1 = nn.Linear(bottleneck, n9)
        self.d2 = nn.Linear(n9, n8)
        self.d3 = nn.Linear(n8, n7)
        self.d4 = nn.Linear(n7, n6)
        self.d5 = nn.Linear(n6, n5)
        self.d6 = nn.Linear(n5, n4)
        self.d7 = nn.Linear(n4, n3)
        self.d8 = nn.Linear(n3, n2)
        self.d9 = nn.Linear(n2, n1)

    def encode(self, x):
        h1 = F.relu(self.e1(x))
        h2 = F.relu(self.e2(h1))
        h3 = F.relu(self.e3(h2))
        h4 = F.relu(self.e4(h3))
        h5 = F.relu(self.e5(h4))
        h6 = F.relu(self.e6(h5))
        h7 = F.relu(self.e7(h6))

        mu = self.e8_1(h7)
        logvar = self.e8_2(h7)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        H1 = F.relu(self.d1(z))
        H2 = F.relu(self.d2(H1))
        H3 = F.relu(self.d3(H2))
        H4 = F.relu(self.d4(H3))
        H5 = F.relu(self.d5(H4))
        H6 = F.relu(self.d6(H5))
        H7 = F.relu(self.d7(H6))
        H8 = F.relu(self.d8(H7))
        x_h = torch.sigmoid(self.d9(H8))
        return x_h

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, n1))

        # To remove variational aspects, just decode mu
        if self.AE is True:
            z = mu
        else:
            # Otherwise, perform re-parameterization
            z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

    def set_beta(self, new_beta):
        self.beta = new_beta


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta, AE, ss=False):
    """Caclulates the loss.

    Inputs
    ----------
    recon_x : tensor N, n_features
        Reconstructed signals

    x : tensor N, n_features
        Original signals
    
    mu : tensor N, bottleneck size
        latent representation of original signals    

    logvar : tensor N, bottleneck size
        Logaritmised variables
        
    Outputs:
    ----------
    loss : tensor
        Calculated loss
        
    bce_log : numpy array
        BCE contribution to loss

    kld_log : numpy array
        KLD contribution to loss
    """

    recon_x = recon_x.reshape(x.shape)

    # Reconstruction loss
    BCE = F.mse_loss(recon_x.reshape(-1, np.product(x.shape)),
                     x.reshape(-1, np.product(x.shape)),
                     reduction='sum')/x.shape[0]

    if AE is True:
        # Just the usual mse loss
        return BCE
    
    else:
        # Force regularization on latent space

        # Regularization loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/x.shape[0]

        if KLD == np.inf:
            KLD = torch.tensor(1e12)
        
        # Included for logging
        bce_log = BCE.detach().cpu()
        kld_log = beta*KLD.detach().cpu()

        return BCE + KLD*beta, bce_log, kld_log


def calc_distance(xy):
    """Caclulates the center of gravity (cog) and average distance to cog
    for all points

    Inputs
    ----------
    xy : tensor N, n_features
        Coordinates of datapoints

    Outputs:
    ----------
    cog : tensor 1, n_features
        Average position, i.e. center of gravity
    intra_cd : tensor, float
        Average distance to cog
    """

    # Center of gravity
    cog = xy.mean(axis=0)
    
    # Average distance to cog
    cogv = cog.repeat(len(xy), 1)
    intra_cd = torch.pairwise_distance(xy, cogv).sum()

    return cog, intra_cd


def calc_ss_loss(mu, targets):
    cluster_centers = []
    intra_distances = []

    labels = targets.unique()

    n = len(targets)
    # Loop over each labelled species
    for k in range(len(labels)):
        label = labels[k]

        coords = mu[targets == label]
        nk = len(coords)

        cluster_center, intra_cd = calc_distance(coords)

        # Store results
        cluster_centers.append(cluster_center)
        intra_distances.append(intra_cd*nk/n)

    # Distance between clusters - Should be big
    _, inter_distance = calc_distance(torch.stack(cluster_centers))
    inter_distance = inter_distance/len(labels)

    # Distance within clusters - Should be small
    tot_intra_distances = torch.stack(intra_distances).sum()

    # Cluster ratio, small/big
    CR = (tot_intra_distances + 1E-6)/(inter_distance + 1E-6)

    return inter_distance, tot_intra_distances, CR


def check_for_nan(input_tensor):
    return torch.isnan(input_tensor).any()


def train(epoch):
    model.train()
    train_loss = 0
    BCE_loss = 0
    KLD_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE, KLD = loss_function(recon_batch, data, mu,
                                       logvar, model.beta, model.AE)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        for param in model.parameters():
            if check_for_nan(param):
                print('Nan values in weights!!')
                return [recon_batch, data, mu, logvar, model.beta, model.AE]

        BCE_loss += BCE.item()
        KLD_loss += KLD.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    n_data = len(train_loader.dataset)
    print('====> Average loss: {:.6f}'.format(
          train_loss/len(train_loader.dataset)))
    print('====> Average BCE loss: {:.6f}'.format(
          BCE_loss/len(train_loader.dataset)))
    print('====> Average KLD loss: {:.6f}'.format(
          KLD_loss/len(train_loader.dataset)))

    if model.epsilon > 0:
        ss = 0
        for batch_idx, (data, targets) in enumerate(ss_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, BCE, KLD = loss_function(recon_batch, data, mu,
                                           logvar, model.beta, model.AE)

            dist_info = calc_ss_loss(mu, targets)
            ss_loss = model.epsilon*dist_info[-1]

            if check_for_nan(ss_loss):
                print('SS: Nan values in ss_loss!')
                return [recon_batch, data, mu, logvar, model.beta, model.AE]

            ss += ss_loss.item()
            loss = loss + ss_loss
            # loss = ss_loss
            loss.backward()
            optimizer.step()

        print('====> Average ss loss: {:.6f}'.format(
              loss/len(ss_loader.dataset)))

        return train_loss/n_data, BCE_loss/n_data, KLD_loss/n_data, ss/len(ss_loader.dataset)
    else:
        return train_loss/n_data, BCE_loss/n_data, KLD_loss/n_data, 0


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            loss, BCE, KLD = loss_function(recon_batch, data, mu,
                                           logvar, model.beta, model.AE)

            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 10)
                original = data[:n]
                reconstructed = recon_batch.view(batch_size,
                                                 data_shape[0],
                                                 data_shape[1])[:n]

                if epoch % 5 == 0:
                    compare_reconstructed(original.cpu(),
                                          reconstructed.cpu(),
                                          epoch)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, BCE, KLD = loss_function(recon_batch, data, mu,
                                           logvar, model.beta, model.AE)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)
    print('====> Validation loss: {:.4f}'.format(val_loss))
    return val_loss


def validate_ss(epoch):
    model.eval()

    ss = 0
    for batch_idx, (data, targets) in enumerate(ss_val_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE, KLD = loss_function(recon_batch, data, mu,
                                       logvar, model.beta, model.AE)

        dist_info = calc_ss_loss(mu, targets)

        # Unscaled, since its only used for validation
        ss_loss = dist_info[-1]

        ss += ss_loss.item()

    ss /= len(ss_val_loader.dataset)

    print('====> Validation loss: {:.4f}'.format(ss))
    return ss


# %% Plotting functions
def plot_spectra(event, ax, n=145, x=False, scale=1, alpha=1, lw=1,
                 autoscale=True, colors=False):
    """Plots an event and returns the axis"""

    try:
        event = np.array(event.detach().cpu())
    except Exception:
        pass

    event = event[:, :n]

    if colors is False:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey']

    if x is False:
        x = np.linspace(0, 2000, n)

    lines = []
    for j in range(len(event)):
        lines.append(plt.Line2D(x, event[j, :]*scale,
                                color=colors[j],
                                lw=lw, alpha=alpha))

    for line in lines:
        ax.add_artist(line)

    if autoscale is True:
        xys = np.array([x, event[0, :]]).T
        ax.update_datalim(xys)
        ax.autoscale()

    return ax


def scatterline(data, positions, ax=False, color=None):
    """Create a scatterplot, with a plot of the events rather than
       just a dot"""

    if ax is False:
        fig, ax = plt.subplots()

    if color is None:
        colors = cm.rainbow(np.linspace(0, 1, 2))
    else:
        colors = [color, color]

    for i in range(len(data)):
        array = data[i]
        xy = positions[i]

        # Create drawing area
        da = ob.DrawingArea(10, 10, clip=False)

        # Weird xaxis, has to be roughly same as drawing area size?
        x = np.linspace(0, 10, 145)

        plot_spectra(array, da, x=x, lw=0.5, alpha=0.5,
                     autoscale=False, scale=10, colors=colors)
        # Add drawing area to weird box thingy...
        ab = ob.AnnotationBbox(da, xy, xycoords='data', frameon=False)
        # Add weird box thing to axis object
        ax.add_artist(ab)

    # Autoscale axes
    ax.update_datalim(positions)
    ax.autoscale()

    return ax.get_figure()


def vizualise_latent_space(epoch, dataloader=val_loader, label=None,
                           close=True, color='steelblue',
                           save=True, fig_dots=None, fig_lines=None,
                           drawscatterline=True, threeD=False,
                           title=False):
    global model
    global path

    # For most cases
    if bottleneck == 2:
        if fig_dots is not None:
            ax_d = fig_dots.get_axes()[0]
        else:
            fig_dots, ax_d = plt.subplots(1)

        if drawscatterline is True:
            if fig_lines is not None:
                ax_l = fig_lines.get_axes()[0]
            else:
                fig_lines, ax_l = plt.subplots(1)
    else:
        drawscatterline = False

    # Special 3d case
    if (bottleneck >= 3):
        if (threeD is True) & (bottleneck == 3):
            if fig_dots is None:
                fig_dots = plt.figure(figsize=[9, 9])
                ax_d = fig_dots.add_subplot(111, projection='3d')
                ax_d.set_xlabel('latent dim 0')
                ax_d.set_ylabel('latent dim 1')
                ax_d.set_zlabel('latent dim 2')

        # For higher dimensions, one histogram per dimension
        else:
            if fig_dots is None:
                fig_dots, axes = plt.subplots(bottleneck, figsize=[9, 9],
                                              sharex=True, sharey=True)

    # Generate coordinates
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)

        positions = mu.detach().cpu()

        # Ordinary, 2d scatterplot
        if bottleneck == 2:
            ax_d.scatter(positions[:, 0], positions[:, 1],
                         c=labels, alpha=1, s=5)

        else:
            # 3d plot, if desired
            if (bottleneck == 3) & threeD:
                ax_d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c=color, alpha=0.1, s=5)

            # Any other case -> histograms
            else:
                for i in range(bottleneck):
                    hist, bins = np.histogram(positions[:, i], bins='auto')
                    axes[i].plot(bins[1:], hist, color=color)

        # Draw scatterlines, only 2d is supported
        if drawscatterline is True:
            fig_lines = scatterline(np.array(data.detach().cpu()),
                                    np.array(mu.detach().cpu()), ax=ax_l,
                                    color=color)

    if title is not False:
        fig.suptitle(title)

    # Save figure to disk
    if save is True:
        fig_dots.savefig(path + 'latent spaces/latent_' + str(epoch), dpi=dpi)
        if drawscatterline is True:
            fig_lines.savefig(path + 'latent spaces/latent_cool_' + str(epoch),
                              dpi=dpi)

    # Close figures
    if close is True:
        plt.close(fig_dots)
        plt.close(fig_lines)


def compare_reconstructed(original, reconstructed, epoch,
                          save=True, close=True):
    assert original.shape == reconstructed.shape
    n = original.shape[0]

    fig, axes = plt.subplots(n, 2, sharey=True)
    for i in range(n):
        plot_spectra(original[i, :, :], axes[i, 0])
        plot_spectra(reconstructed[i, :, :], axes[i, 1])

    for ax in axes.flatten():
        ax.set_xticks(np.linspace(0, 1500, 6))
        ax.set_xticklabels([])
        ax.grid()
        ax.set_yticks([])

    for ax in axes[n-1, :]:
        ax.set_xticklabels(np.linspace(0, 1500, 6).astype(int))

    if save is True:
        fig.savefig(path + 'comparisons/comparison_' + str(epoch), dpi=dpi)

    if close is True:
        plt.close(fig)


# %% Pave the road
def pave_the_road(folder_name):
    global master_path

    comparisons = master_path + folder_name + 'comparisons/'
    latent = master_path + folder_name + 'latent spaces/'
    loss_curves = master_path + folder_name + 'loss curves/'
    if os.path.isdir(master_path + folder_name) is False:
        os.mkdir(master_path + folder_name)
        os.mkdir(comparisons)
        os.mkdir(latent)
        os.mkdir(loss_curves)
    else:
        for file in os.listdir(comparisons)[::-1]:
            os.remove(comparisons + file)
        for file in os.listdir(latent)[::-1]:
            os.remove(latent + file)
        for file in os.listdir(loss_curves)[::-1]:
            os.remove(loss_curves + file)

    return master_path + folder_name


# %% Train network
plt.close('all')

# Create / empty output folderes
path = pave_the_road(folder_name)

# Initialize network
model = VAE(beta=0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.epsilon = 0

# Logging parameters
train_loss_list = []
val_loss_list = []
test_loss_list = []
ss_loss_list = []

beta_list = []
epsilon_list = []

Lrec = []
Lreg = []

epochs = []
delta_rec_list = []
delta_reg_list = []
delta_Lrec_list = []
term_1_list = []
term_2_list = []
changes = []

# %% Main loop
epoch = 0
last_change = 0
for epoch in range(epoch, 1000):
    # Emergency break, to keep variables available for debugging
    output = train(epoch)

    if len(output) == 4:
        tr_loss, bce_loss, kld_loss, ss_loss = output
    else:
        recon_x, x, mu, logvar, model.beta, model.AE = output
    assert len(output) == 4

    val_loss = validate_ss(epoch)
    te_loss = test(epoch)

    # Log loss values
    train_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    test_loss_list.append(te_loss)
    ss_loss_list.append(ss_loss)
    
    Lrec.append(bce_loss)
    beta_list.append(model.beta)
    epsilon_list.append(model.epsilon)

    if model.beta > 0:
        Lreg.append(kld_loss/model.beta)
    else:
        Lreg.append(np.nan)
    print('Beta:', model.beta)

    title = str(model.beta)
    if epoch % 5 == 0:
        # Visualize latent space
        vizualise_latent_space(epoch, close=True, drawscatterline=False,
                               dataloader=ss_loader)

    if epoch % 10 == 0:
        # Visualize latent space
        vizualise_latent_space(epoch, close=True, drawscatterline=False,
                               dataloader=val_loader)

    # Adjust the beta term
    if epoch == 24:
        model.beta = 1e-3
        
    # Get the ball rolling before fiddeling about
    if (model.beta < max_beta) & (epoch > 25):
        if vanilla is True:
                model.beta = 1
        else:
            if (epoch - last_change) > 10:
                delta_rec = Lrec[-1] - w1*np.min(Lrec[25:])
                delta_reg = Lreg[-1] - w2*np.min(Lreg[25:])
                
                # Small check
                delta_Lrec = np.sign(Lrec[-1] - w3*Lrec[last_change]) + np.sign(Lrec[-1] - w4*Lrec[last_change])
            
                # Decrease beta term
                term_1 = (b/4)*(1 - np.sign(delta_reg))*(1 + np.sign(delta_rec) + delta_Lrec)
            
                # Increase beta term
                term_2 = (a/4)*(1 - np.sign(delta_rec))*((1 - np.sign(delta_reg) - delta_Lrec))
            
                model.beta = model.beta - term_1 + term_2
                if term_2 - term_1 != 0:
                    last_change = epoch
                    changes.append(epoch)
                    
                epochs.append(epoch)
                delta_rec_list.append(delta_rec)
                delta_reg_list.append(delta_reg)
                delta_Lrec_list.append(delta_Lrec)
                term_1_list.append(term_1)
                term_2_list.append(term_2)
    
    			# Never allow beta to go negative!
                if model.beta <= 0:
                    model.beta = 0.05
						
        if semi_supervised is True:
            # Supervised clustering loss
            if (epoch > 50) & (model.epsilon == 0):
                model.epsilon = 0.01
        
            if (epoch >= 100) & (epoch % 100 == 0):
                if model.epsilon == 0.2:
                    model.epsilon = 0.01
                else:
                    model.epsilon = 0.2

    # Post training tasks
    # Plot loss curves
    if epoch % 5 == 0:
        xmin = 0
        x = np.arange(xmin, len(train_loss_list))
        plt.close('all')
        fig, axes = plt.subplots(4, sharex=True, figsize=[36, 18])
        if epoch > 500:
            axes[0].set_xlim([epoch-500, epoch])
            axes[0].set_ylim([0, np.mean(train_loss_list[-1000:])*1.1])

        # Plot total loss
        axes[0].plot(x, train_loss_list[xmin:],
                     color='steelblue', label='Training loss - Total')

        # Lrec
        axes[0].plot(x, Lrec[xmin:],
                     color='steelblue', ls='--', label='$L_{rec}$')

        # Lreg, with beta scaling
        axes[0].plot(x, np.array(Lreg[xmin:])*np.array(beta_list[xmin:]),
                     color='red', ls=':',
                     label='$\\beta L_{reg}$')

        # Lreg, without beta scaling
        axes[0].plot(x, Lreg[xmin:],
                     color='red', ls='--',
                     label='$L_{reg}$')

        # Semi Supervised Loss, with gamma scaling
        axes[0].plot(x, ss_loss_list[xmin:],
                     color='green', ls=':',
                     label='$L_{cls}$')

        # Plot beta
        axes[1].plot(x, beta_list[xmin:], label='beta', color='red')
        axes[1].plot(x, epsilon_list[xmin:], label='epsilon', color='orange')

        # Plot delta rec
        axes[2].plot(np.array(epochs)[xmin:],
                      np.array(delta_rec_list)[xmin:],
                      label='$\Delta_{rec}$', color='Blue')

       # Plot delta reg
        axes[2].plot(np.array(epochs)[xmin:],
                      np.array(delta_reg_list)[xmin:],
                      label='$\Delta_{reg}$', color='Red')

       # Plot delta Lrec
        axes[3].plot(np.array(epochs)[xmin:],
                      np.array(delta_Lrec_list)[xmin:],
                      label='$\Delta L_{rec}$', color='Green')

        axes[3].plot(np.array(epochs)[xmin:],
                     np.sign(delta_rec_list[xmin:]),
                     label='$\psi[\Delta_{rec}]$', color='Blue')
        axes[3].plot(np.array(epochs)[xmin:],
                     np.sign(delta_reg_list[xmin:]),
                     label='$\psi[\Delta_{reg}]$', color='Red')

        # Make plots pretty
        axes[0].legend(loc=3, ncol=3)
        axes[0].set_ylabel('Loss')

        axes[3].set_xlabel('Number of epochs')

        for ax in axes:
            # ax.set_xticks(np.array(checks)[xmin:, 0])
            ax.grid()
            ax.legend()
        fig.savefig(path + 'loss curves/' + 'Loss curves ' + str(epoch))

# Save model
torch.save(model.state_dict(), path + 'model.pt')

