import itertools
import torch
from trainer import Trainer
from models import BERTForQuantification
from datasets import load_dataset
from transformers import BertModel, RobertaModel, AlbertModel, ElectraModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

def calculate_metrics(targets, predictions):
    """
    Calculate regression metrics: MAE, MSE, RMSE, R², and MAPE.
    """
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(targets, predictions)
    mape = (torch.abs((torch.tensor(targets) - torch.tensor(predictions)) / torch.tensor(targets))).mean().item() * 100
    return mae, mse, rmse, r2, mape


def grid_search(param_grid, device, save_path="trained_models"):
    """
    Perform grid search over the specified parameter grid, train models, and save the best model.
    """
    best_val_loss = float('inf')
    best_params = None
    best_metrics = None  # To store the best metrics

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

        # Evaluate validation metrics
        all_targets = []
        all_predictions = []
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                score = batch['score'].to(device)
                class_type = batch['class_type']

                outputs = []
                for i in range(len(input_ids)):
                    output = model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0), class_type[i])
                    outputs.append(output)

                outputs = torch.cat(outputs).squeeze(1)
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(score.cpu().numpy())

        mae, mse, rmse, r2, mape = calculate_metrics(all_targets, all_predictions)
        print(f"Validation Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")

        # Save the trained model
        model_save_path = f"{save_path}/{model_name}_lr{lr}_bs{batch_size}_epochs{epochs}_wd{weight_decay}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

        # Update best parameters if current model is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_metrics = (mae, mse, rmse, r2, mape)  # Save the best metrics

        print(f"Finished training. Validation loss: {val_loss}")

    # Final report
    print(f"\nBest Validation Loss: {best_val_loss} with parameters: {best_params}")
    print(f"Best Metrics - MAE: {best_metrics[0]:.4f}, MSE: {best_metrics[1]:.4f}, RMSE: {best_metrics[2]:.4f}, R²: {best_metrics[3]:.4f}, MAPE: {best_metrics[4]:.4f}")

# Define parameter grid
param_grid = {
    'model_name': ['bert-base-uncased','roberta-base', 'albert-base-v2', 'electra-base-discriminator'],
    'learning_rate': [2e-5],
    'batch_size': [16],
    'epochs': [5],
    'weight_decay': [0.01]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device} ')
grid_search(param_grid, device, save_path="trained_models")
