import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from client import Client
from server import Server
from environment import FL_Environment
from dqn_agent import DQN_Agent
from torch.utils.data import Subset,DataLoader
import matplotlib.pyplot as plt
import random

# Simulate Homogeneous Data

# Parameters
num_clients = 5
k = 3 #per round clients
num_classes = 10  # Number of classes in dataset

# Load dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

#mnist transforms
transform_train_mnist = transforms.Compose([
    transforms.RandomRotation(10),  # Slight rotation for data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
])

transform_test_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

#cifar transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train_mnist)
mnist_test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test_mnist)

cifar_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Assigning the dataset
dataset = cifar_dataset

# Shuffle indices
total_samples = len(dataset)
indices = np.arange(len(dataset))
np.random.shuffle(indices)

# Divide dataset equally among clients
client_datasets = []
samples_per_client = len(dataset) // (num_clients)

"""
for i in range(num_clients+1):
    subset_indices = indices[i * samples_per_client : (i + 1) * samples_per_client]
    client_datasets.append(Subset(dataset, subset_indices))

"""
# Simulate Heterogeneous Data
start = 0
# Generate unequal data splits for each client
for i in range(num_clients):
    # Random proportion of the dataset for this client
    proportion = np.random.uniform(0.1, 0.2)
    num_samples = int(proportion * total_samples)
    num_samples = min(num_samples, total_samples - start)  # handle leftover

    subindices = indices[start:start + num_samples]
    subset = Subset(dataset, subindices)
    client_datasets.append(subset)

    start += num_samples
    if start >= total_samples:
        break

clients = [Client(i, client_datasets[i]) for i in range(num_clients)]
server = Server()
env = FL_Environment(num_clients, global_class_dist=np.ones(10) / 10)
agent = DQN_Agent(state_size=10, action_size=num_clients)
rewards = []
accuracies = []
participation_freq = np.zeros(num_clients)

# Switch to True if testing for FedAvg
fedAvgTest = False

# Train Federated Model
prev_acc = 0.0
for epoch in range(10):
    print(f"--- Round {epoch + 1} ---")

    # Select clients using RL agent
    #selected_client_indexes = [i for i in agent.select_clients(np.ones(10) / 10, num_clients)]

    if fedAvgTest:
        #selected_client_idxs = list(range(num_clients))
        selected_client_idxs = random.sample(range(num_clients), k)
    else:
        state = env.get_state()
        selected_client_idxs = agent.select_clients(state, num_clients, k)

    selected_clients = [clients[i] for i in selected_client_idxs]

    # Train each selected client
    for client in selected_clients:
        client.train() # Clients train locally
        participation_freq[client.client_id] += 1

    # Aggregate client models into the global model
    server.aggregate_models([client.model for client in selected_clients])

    # Evaluate global model performance
    new_acc, loss = server.evaluate(cifar_test_dataset)  
    #acc, loss = server.evaluate()
    print(f"Global Model - Accuracy: {new_acc:.4f}, Loss: {loss:.4f}")

    # Compute reward and train RL agent
    client_distributions = [client.get_class_distribution() for client in clients]
    #reward = env.step(selected_client_indexes, prev_acc, new_acc, [c.get_class_distribution() for c in clients])
    reward = env.step(
        selected_client_idxs,
        prev_acc,
        new_acc,
        client_distributions,
        participation_freq,
        [len(c.local_data) for c in clients]
    )
    rewards.append(reward)
    accuracies.append(new_acc)
    print(f"Test Accuracy: {new_acc:.4f}")
    if not fedAvgTest:
        agent.train(state, selected_client_idxs[0], reward, env.get_state())
    prev_acc = new_acc

    print(f"Epoch {epoch + 1}, Selected Clients {selected_client_idxs}, Client Selection Loss: {reward:.4f}, Accuracy: {new_acc:.4f}")



#epochs = np.arange(1, len(rewards) + 1)

# Plot loss vs. epochs
#plt.plot(epochs, rewards, marker='o', linestyle='-', color='b', label="Reward")

# Labels and title
#plt.xlabel("Epochs")
#plt.ylabel("Client Selection Loss")
#plt.title("Training client selection loss over Epochs")
#plt.legend()

# Show the graph
#plt.grid(True)  # Optional: Adds grid for better readability
#plt.savefig("./reward_plot.png")



epochs = np.arange(1,len(accuracies)+1)
# Plot loss vs. epochs
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='r', label="Accuracy")

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test accuracy over Epochs")
plt.legend()

# Show the graph
plt.grid(True)  # Optional: Adds grid for better readability
plt.savefig("./accuracy_plot.png")
