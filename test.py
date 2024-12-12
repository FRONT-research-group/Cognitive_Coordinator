import torch
from transformers import BertTokenizer
from models import BERTForQuantification

def load_model(checkpoint_path, device):
    """
    Loads the fine-tuned model from a saved checkpoint.
    """
    # Initialize the model architecture
    model = BERTForQuantification()

    # Load the trained weights directly
    model.load_state_dict(torch.load(checkpoint_path))

    # Set the model to evaluation mode
    model.eval()
    
    model.to(device)

    return model

def tokenize_input(text, tokenizer, max_len=128):
    tokens = tokenizer(text, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
    return tokens['input_ids'], tokens['attention_mask']

def inference(text, class_type, model, tokenizer, device):
    input_ids, attention_mask = tokenize_input(text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        score = model(input_ids, attention_mask, class_type)
    return score.item()

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('output/final_bert_model.pth', device)

    #input_text = "I want secure online meeting"
    #input_text = "I want my data to be secured"
    #input_text = "I want my data not be leaked"
    input_text = "I want to create an app and I want it to be stable"
    trust_functions = {"Safety", "Security", "Reliability", "Resilience", "Privacy"}

    for tf in trust_functions:
        score = inference(input_text, tf, model, tokenizer, device) * 100
        print(f"{tf} Score: {score}")