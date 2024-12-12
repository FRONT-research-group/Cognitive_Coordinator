import itertools
import torch
from trainer import Trainer
from models import BERTForQuantification
from datasets import load_dataset
from transformers import BertModel, RobertaModel, AlbertModel, ElectraModel

def load_pretrained_model(model_name, device):
    """
    Load the specified pre-trained model and return the BERTForQuantification wrapper.
    """
    if model_name == 'bert-base-uncased':
        pretrained_model_name = 'bert-base-uncased'
    elif model_name == 'roberta-base':
        pretrained_model_name = 'roberta-base'
    elif model_name == 'albert-base-v2':
        pretrained_model_name = 'albert-base-v2'
    elif model_name == 'electra-base-discriminator':
        pretrained_model_name = 'google/electra-base-discriminator'
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Pass the model name, not the initialized object
    return BERTForQuantification(pretrained_model_name).to(device)


def grid_search(param_grid, device, save_path="trained_models"):
    """
    Perform grid search over the specified parameter grid, train models, and save the best model.
    """
    best_val_loss = float('inf')
    best_params = None

    for params in itertools.product(*param_grid.values()):
        model_name, lr, batch_size, epochs, weight_decay = params

        print(f"\nTraining with model={model_name}, lr={lr}, batch_size={batch_size}, epochs={epochs}, weight_decay={weight_decay}")
        
        train_dataloader, val_dataloader = load_dataset('datasets/dataset.csv', batch_size=batch_size)
        model = load_pretrained_model(model_name, device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        trainer = Trainer(model, train_dataloader, val_dataloader, device, save_dir='output')
        trainer.optimizer = optimizer
        trainer.train(epochs)

        val_loss = trainer.get_validation_loss()

        # Save the trained model
        model_save_path = f"{save_path}/{model_name}_lr{lr}_bs{batch_size}_epochs{epochs}_wd{weight_decay}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

        # Update best parameters if current model is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

        print(f"Finished training. Validation loss: {val_loss}")

    print(f"\nBest Validation Loss: {best_val_loss} with parameters: {best_params}")

# Define parameter grid
param_grid = {
    'model_name': ['roberta-base', 'albert-base-v2', 'electra-base-discriminator'],
    'learning_rate': [2e-5],
    'batch_size': [16],
    'epochs': [5],
    'weight_decay': [0.01]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device} ')
grid_search(param_grid, device, save_path="trained_models")
