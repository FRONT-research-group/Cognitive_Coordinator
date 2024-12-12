# main.py
from trainer import Trainer
from models import BERTForQuantification
from datasets import load_dataset
import torch

def main():
    # Load the dataset from CSV file
    train_dataloader, val_dataloader = load_dataset(file_path='datasets/dataset.csv', batch_size=16, augment=True)

    # Initialize the model
    model = BERTForQuantification()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the trainer and start training
    trainer = Trainer(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device, save_dir='output')
    trainer.train(epochs=5)

    # Optionally save the final model after training is complete
    torch.save(model.state_dict(), "output/final_bert_model.pth")
    print("Final model saved at output/final_bert_model.pth")

if __name__ == "__main__":
    main()
