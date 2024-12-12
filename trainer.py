import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, save_dir):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = save_dir
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.criterion = nn.MSELoss()

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            running_loss = 0.0
            all_targets = []
            all_predictions = []

            for batch in tqdm(self.train_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                score = batch['score'].to(self.device)
                class_type = batch['class_type']

                self.optimizer.zero_grad()

                # Forward pass
                outputs = []
                for i in range(len(input_ids)):
                    output = self.model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0), class_type[i])
                    outputs.append(output)
                
                outputs = torch.cat(outputs).squeeze(1)
                loss = self.criterion(outputs, score)
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()

                # Collect predictions and targets for metrics
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(score.cpu().numpy())

            # Calculate metrics for training
            mae, mse, rmse, r2, mape = self.calculate_metrics(all_targets, all_predictions)
            print(f"Training loss: {running_loss / len(self.train_dataloader)}")
            print(f"Training Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")

            self.validate()

    def validate(self):
        self.model.eval()
        running_val_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                score = batch['score'].to(self.device)
                class_type = batch['class_type']

                outputs = []
                for i in range(len(input_ids)):
                    output = self.model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0), class_type[i])
                    outputs.append(output)
                
                outputs = torch.cat(outputs).squeeze(1)
                val_loss = self.criterion(outputs, score)
                running_val_loss += val_loss.item()

                # Collect predictions and targets for metrics
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(score.cpu().numpy())

        # Calculate metrics for validation
        mae, mse, rmse, r2, mape = self.calculate_metrics(all_targets, all_predictions)
        print(f"Validation loss: {running_val_loss / len(self.val_dataloader)}")
        print(f"Validation Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")

        self.final_val_loss = running_val_loss / len(self.val_dataloader)

    def calculate_metrics(self, targets, predictions):
        """
        Calculate regression metrics: MAE, MSE, RMSE, R², and MAPE.
        """
        mae = mean_absolute_error(targets, predictions)
        mse = ((torch.tensor(targets) - torch.tensor(predictions)) ** 2).mean().item()
        rmse = mse ** 0.5
        r2 = r2_score(targets, predictions)
        mape = torch.mean(torch.abs((torch.tensor(targets) - torch.tensor(predictions)) / torch.tensor(targets))).item() * 100
        return mae, mse, rmse, r2, mape

    def get_validation_loss(self):
        """
        Returns the final validation loss after training.
        """
        return self.final_val_loss

    def save_model(self, epoch):
        """
        Save the model state dictionary and training epoch for checkpointing.
        """
        save_path = f"{self.save_dir}/bert_model_epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

