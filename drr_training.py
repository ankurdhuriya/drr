from copyreg import pickle
import os
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from datetime import datetime
from model.model import Actor, Critic, DRRAveStateRepresentation, PMF
from learn.learn import DRRTrainer

import matplotlib.pyplot as plt
from tsmoothie.smoother import ConvolutionSmoother

class config:
    date_time = datetime.now().strftime('%y%m%d-%H%M%S')
    output_path = 'results/' + date_time + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plot_dir = output_path + 'rewards.pdf'
 
    train_actor_loss_data_dir = output_path + 'train_actor_loss_data.npy'
    train_critic_loss_data_dir = output_path + 'train_critic_loss_data.npy'
    train_mean_reward_data_dir = output_path + 'train_mean_reward_data.npy'
 
    train_actor_loss_plot_dir = output_path + 'train_actor_loss.png'
    train_critic_loss_plot_dir = output_path + 'train_critic_loss.png'
    train_mean_reward_plot_dir = output_path + 'train_mean_reward.png'
 
    trained_models_dir = 'trained/' + date_time + '/'
 
    actor_model_trained = trained_models_dir + 'actor_net.weights'
    critic_model_trained = trained_models_dir + 'critic_net.weights'
    state_rep_model_trained = trained_models_dir + 'state_rep_net.weights'
 
    actor_model_dir = output_path + 'actor_net.weights'
    critic_model_dir = output_path + 'critic_net.weights'
    state_rep_model_dir = output_path + 'state_rep_net.weights'
 
    csv_dir = output_path + 'log.csv'
 
    path_to_trained_pmf = 'trained/trained_pmf.pt'
 
    # hyperparams
    batch_size = 128
    gamma = 0.9
    replay_buffer_size = 100000
    history_buffer_size = 5
    learning_start = 1000 #5000
    learning_freq = 1
    lr_state_rep = 0.001
    lr_actor = 0.0001
    lr_critic = 0.001
    eps_start = 1
    eps = 0.1
    eps_steps = 10000
    eps_eval = 0.1
    tau = 0.01 # inital 0.001
    beta = 0.4
    prob_alpha = 0.3
    max_timesteps_train = 5000
    max_epochs_offline = 500
    max_timesteps_online = 2000
    embedding_feature_size = 100
    episode_length = 10
    train_ratio = 0.8
    weight_decay = 0.01
    clip_val = 1.0
    log_freq = 500
    saving_freq = 100
    zero_reward = False
 
    no_cuda = True
    
    logs_dir = 'runs/' + date_time

def seed_all(cuda, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed=seed)

print("Initializing DRR Framework ----------------------------------------------------------------------------")
 
# Get CUDA device if available
cuda = True if not config.no_cuda and torch.cuda.is_available() else False
print("Using CUDA") if cuda else print("Using CPU")
 
# Init seeds
seed_all(cuda, 0)
print("Seeds initialized")
 
# Grab models
actor_function = Actor
critic_function = Critic
state_rep_function = DRRAveStateRepresentation

CSV_PATH = 'dataset/sample_data.csv'
data_df = pd.read_csv(CSV_PATH)
reward_map = {'view': 1, 'cart': 2, 'purchase': 3}
data_df['behavior'] = data_df['event_type'].apply(lambda x : reward_map[x])

with open('dataset/user_num_to_id.pkl', 'rb') as f:
    users = pickle.load(f)

with open('dataset/item_num_to_id.pkl', 'rb') as f:
    items = pickle.load(f)

NUM_USERS, NUM_ITEMS = len(users), len(items)

data = data_df.loc[:, ['user_id_num', 'product_id_num', 'behavior', 'event_time']].values

shuffle(data, random_state=1)
train_data = torch.from_numpy(data[:int(config.train_ratio * data.shape[0])])
test_data = torch.from_numpy(data[int(config.train_ratio * data.shape[0]):])
print("Data imported, shuffled, and split into Train/Test, ratio=", config.train_ratio)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)
 
# Create and load PMF function for rewards and embeddings
reward_function = PMF(NUM_USERS, NUM_ITEMS, config.embedding_feature_size, is_sparse=False, no_cuda=~cuda)
reward_function.load_state_dict(torch.load(config.path_to_trained_pmf))
 
# Freeze all the parameters in the network
for param in reward_function.parameters():
    param.requires_grad = False
print("Initialized PMF, imported weights, created reward_function")
 
# Extract embeddings
user_embeddings = reward_function.user_embeddings.weight.data
item_embeddings = reward_function.item_embeddings.weight.data
print("Extracted user and item embeddings from PMF")
print("User embeddings shape: ", user_embeddings.shape)
print("Item embeddings shape: ", item_embeddings.shape)
 
# Init trainer
print("Initializing DRRTrainer -------------------------------------------------------------------------------")
trainer = DRRTrainer(config,
                      actor_function,
                      critic_function,
                      state_rep_function,
                      reward_function,
                      users,
                      items,
                      train_data,
                      test_data,
                      user_embeddings,
                      item_embeddings,
                      cuda
                      )

# Train
print("Starting DRRTrainer.learn() ---------------------------------------------------------------------------")
actor_losses, critic_losses, epi_avg_rewards = trainer.learn()

def noiseless_plot(y, title, ylabel, save_loc):
    smoother = ConvolutionSmoother(window_len=1000, window_type='ones')
    smoother.smooth(y)

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=3)

    # plot the smoothed timeseries with intervals
    plt.close()
    plt.figure(figsize=(11,6))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(smoother.data[0], color='orange')
    plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
    plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)
    plt.savefig(save_loc)
    plt.close()

actor_losses = np.load(config.train_actor_loss_data_dir)
critic_losses = np.load(config.train_critic_loss_data_dir)
epi_avg_rewards = np.load(config.train_mean_reward_data_dir)

noiseless_plot(actor_losses, 
               "Actor Loss (Train)", 
               "Actor Loss (Train)", 
               config.output_path + "train_actor_loss_smooth.png")
               
noiseless_plot(critic_losses, 
               "Critic Loss (Train)", 
               "Critic Loss (Train)", 
               config.output_path + "train_critic_loss_smooth.png")

noiseless_plot(epi_avg_rewards, 
               "Mean Reward (Train)", 
               "Mean Reward (Train)", 
               config.output_path + "train_mean_reward_smooth.png")