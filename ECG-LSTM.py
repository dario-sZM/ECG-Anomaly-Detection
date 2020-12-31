import torch
import copy
import numpy as np 
import pandas as pd 
import seaborn as sns 
from pylab import rcParams
import matplotlib.pyplot as plt 
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim
import torch.nn.functional as F
from arff2pandas import a2p

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Assign the torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Assigned Torch')

# Load data from arff file
with open('./ECG5000/ECG5000_TRAIN.arff') as f:
    train = a2p.load(f)

with open('./ECG5000/ECG5000_TEST.arff') as f:
    test = a2p.load(f)

# Concatenate train and test data to provide larger base for learning
ecg_df = train.append(test)
print('ECG arff file loaded | Concatenated')

CLASS_NORMAL = 1
class_names = ['Normal', 'R on T', 'PVC', 'SP or EB', 'UB']

# Rename the final column in the ECG data to make more sense
new_columns = list(ecg_df.columns)
new_columns[-1] = 'Heartbeat Classes'
ecg_df.columns = new_columns

#Shuffle the data rows to avoid patterns
ecg_df = ecg_df.sample(frac=1.0)  

# Function to convert to and plot time series data, rolling average with standard deviations
def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
    )
    ax.set_title(class_name)

classes = ecg_df['Heartbeat Classes'].unique()

# Setup suplots and figure
fig, axs = plt.subplots(
    nrows=len(classes) // 3 + 1,
    ncols=3,
    sharey=True,
    figsize=(10, 6)
    )

# Retrieve individual data for each class and use the plotting function from before
for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = ecg_df[ecg_df['Heartbeat Classes'] == cls] \
        .drop(labels='Heartbeat Classes', axis=1) \
        .mean(axis=0) \
        .to_numpy()
    plot_time_series_class(data, class_names[i], ax)

fig.delaxes(axs.flat[-1])
fig.tight_layout()

# Retrieve all the normal ECG recordings
normal_df = ecg_df[ecg_df['Heartbeat Classes'] == str(CLASS_NORMAL)].drop(labels='Heartbeat Classes', axis=1)

# Retrieve all the abnormal ECG recordings (any of the other 4 classes)
anomaly_df = ecg_df[ecg_df['Heartbeat Classes'] != str(CLASS_NORMAL)].drop(labels='Heartbeat Classes', axis=1)

# Split normal data into 85% training and 15% validation data
train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=RANDOM_SEED)

# Furthur split the validation data into 50% validation and 50% test data
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=RANDOM_SEED)

# Convert all data frames into sequences
train_sequences = train_df.astype(np.float32).to_numpy().tolist()
val_sequences = val_df.astype(np.float32).to_numpy().tolist()
test_sequences = test_df.astype(np.float32).to_numpy().tolist()
anomaly_sequences = anomaly_df.astype(np.float32).to_numpy().tolist()

# Function to convert sequences into torch tensors
def create_dataset(sequences):
    dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]

    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features

# Convert sequences into torch tensors
train_dataset, seq_len, n_features = create_dataset(train_sequences)

val_dataset, _, _ = create_dataset(val_sequences)
test_normal_dataset, _, _ = create_dataset(test_sequences)
test_anomaly_dataset, _, _ = create_dataset(anomaly_sequences)

# Encoder 
# The Encoder uses two LSTM layers to compress the Time Series data input.
class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    # Layer 1
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    # Layer 2
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

# Decoder
# Our Decoder contains two LSTM layers and an output layer that gives the final reconstruction.
class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    # Layer 1
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    # Layer 2
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)

# Final LSTM Autoencoder module
# Our Autoencoder passes the input through the Encoder and Decoder.
class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
    
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# Create an instance of the autoencoder module
model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=128)
model = model.to(device)

# Define a function for training
def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    print('Running model...')
    model = model.train()
    
    train_losses = []
    
    for seq_true in train_dataset:
      optimizer.zero_grad()
      
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      
      loss = criterion(seq_pred, seq_true)
      
      loss.backward()
      optimizer.step()
      
      train_losses.append(loss.item())
    
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
        print(f'| EPOCH: {epoch} | TRAIN LOSS: {train_loss} | VAL LOSS: {val_loss} |')
    
    model.load_state_dict(best_model_wts)
    return model.eval(), history

model, history = train_model(
    model, 
    train_dataset,
    val_dataset,
    n_epochs=150
)