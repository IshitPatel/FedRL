import torch
import torch.nn as nn
import torch.optim as optim
from models import CNNModel, ResNetFed
from torch.utils import data
from collections import Counter

class Server:
    """Federated Learning Server."""
    def __init__(self, num_classes = 10):
        #self.global_model = SimpleNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = ResNetFed().to(self.device)

    def aggregate_models(self, client_models):
        """Perform FedAvg aggregation."""
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            tensors = [client_models[i].state_dict()[k].float() for i in range(len(client_models))]
            global_dict[k] = torch.stack(tensors).mean(0)
        
        self.global_model.load_state_dict(global_dict)

        for client in client_models:  
            client.load_state_dict(global_dict)
        

    def evaluate(self, test_loader):
        """Evaluate global model on test data."""
        self.global_model.eval()
        correct, total, loss = 0, 0, 0
        criterion = nn.CrossEntropyLoss()
        test_loader = data.DataLoader(test_loader, batch_size=64, shuffle=False)

        with torch.no_grad():
            #print(len(test_loader))
            for images, labels in test_loader:
                #print(len(images))
                outputs = self.global_model(images)
                #print(len(outputs),outputs)
                loss += criterion(outputs, labels).item()
                #print(Counter(outputs.argmax(dim=1)))
                _,predicted = torch.max(outputs,1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * (correct/total)
        
        return accuracy, loss / len(test_loader)
