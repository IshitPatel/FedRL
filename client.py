import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from models import CNNModel, ResNetFed


class Client:
    """Federated Learning Client."""
    def __init__(self, client_id, dataset, num_classes = 10):
        self.client_id = client_id
        self.local_data = data.DataLoader(dataset, batch_size=128, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetFed().to(self.device)
        #self.model = SimpleNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    def train(self, epochs=5):
        """Train the client's model locally."""
        self.model.train()
        for epoch in range(epochs):
            runningLoss = 0.0
            total, correct = 0, 0
            for images, labels in self.local_data:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            print(f"Client ID: {self.client_id}, Epoch {epoch+1}: Loss: {runningLoss:.4f}, Accuracy: {correct / total:.4f}")
            self.scheduler.step()

        return self.model.state_dict()

    def get_class_distribution(self):
        """Get class distribution in client dataset."""
        class_counts = torch.bincount(torch.tensor([label for _, label in self.local_data.dataset]))
        return class_counts.float() / class_counts.sum()
