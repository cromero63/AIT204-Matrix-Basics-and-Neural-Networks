import json, joblib, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
import string

class NbaTeamSelectorANN(nn.Module):
    """
    This is a ML model that is designed to create an ANN to select 5 optimal team members for and NBA team.
    """
    def __init__(self, input_dim, num_classes):
        """
        Initializes the model
        
        :param input_dim: The input size of the ANN model
        :param num_classes: The output size of the ANN model
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
            nn.Linear(input_dim, 800), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(800, 400), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(400, 100), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(100, num_classes),
            nn.Sigmoid()
        ).to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.crit = nn.BCELoss()

    def forward(self, X):
        """
        Performs a forward pass of data through the ANN
        
        :param X: the input values
        """
        return self.net(X)

    def train_model(self, X, y, epochs=80):
        """
        Trains the ANN model based on the passed in X and y. 
        The epochs is the number of training loops that will be performed.
        
        :param X: The input NumPy array
        :param y: The target NumPy Array
        :param epochs: The number of loops to train on
        """
        # Convert NumPy arrays to Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # train for number of inputed epochs
        for epoch in range(epochs):
            self.train() 
            logits = self(X_tensor)
            loss = self.crit(logits, y_tensor)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X):
        """
        Runs forward propagation and applies a threshold to get 
        the one-hot representation of the optimal 5 players.
        """
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            # 1. Forward propagate
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            probabilities = self.forward(X_tensor)
            
            # 2. Extract Top 5 indices
            # Since we need exactly 5 players, we use topk instead of a 0.5 threshold
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # 3. Create One-Hot Representation
            one_hot = torch.zeros_like(probabilities)
            one_hot.scatter_(1, top_indices, 1.0)
            
        return one_hot.cpu().numpy(), top_indices.cpu().numpy()

    def evaluate(self, X_test, y_test):
        """
        Calculates the loss on a test set to check for overfitting.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y_test, dtype=torch.float32, device=self.device)
            
            predictions = self.forward(X_tensor)
            test_loss = self.crit(predictions, y_tensor)
            
        return test_loss.item()
