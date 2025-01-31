import torch
import torch.nn as nn
import torch.special as special

class MSE(nn.Module):
    """
    Compute mean squared error (MSE)
    """
    
    def __init__(self):
        
        super(MSE, self).__init__()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        nan_mask = ~torch.isnan(targets)
        squared_diff = (predictions[nan_mask] - targets[nan_mask]) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        return mean_squared_error
    
class BCE(nn.Module):
    """
    Compute mean squared error (MSE)
    """
    
    def __init__(self):
        
        super(BCE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        nan_mask = ~torch.isnan(targets)
        loss = self.criterion(predictions[nan_mask], targets[nan_mask])

        return loss
    
class MAE(nn.Module):
    """
    Compute Mean Absolute Error (MAE)
    """
    
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, predictions, targets):
        # Calculate the absolute differences between predictions and targets
        nan_mask = ~torch.isnan(targets)
        absolute_diff = torch.abs(predictions[nan_mask] - targets[nan_mask])
        
        # Calculate the mean absolute error
        mean_absolute_error = torch.nanmean(absolute_diff)
        
        return mean_absolute_error

class RMSE(nn.Module):
    """
    Compute root mean squared error (RMSE)
    """
    
    def __init__(self):
        
        super(RMSE, self).__init__()


    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        nan_mask = ~torch.isnan(targets)
        squared_diff = (predictions[nan_mask] - targets[nan_mask]) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        # Take the square root to get the RMSE
        rmse = torch.sqrt(mean_squared_error)
        
        return rmse