import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer
import random
from nltk.corpus import wordnet

# Define augmentation techniques
def synonym_replacement(text, n=2):
    """
    Replace n random words in the text with their synonyms.
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            if synonym != word:  # Ensure synonym is different
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
    return ' '.join(new_words)

def augment_text(text):
    """
    Apply a random augmentation technique to the input text.
    """
    techniques = [synonym_replacement]  # Add more techniques here if needed
    augmentation = random.choice(techniques)
    return augmentation(text)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        text = sample['Keywords']  # Text input
        class_type = sample['Class']  # Class type: Reliability, Privacy, etc.
        score = sample['Score']  # The actual score as the target

        # Tokenize the text using the provided tokenizer
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")

        # Return the tokenized inputs and the corresponding score and class
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'class_type': class_type,
            'score': torch.tensor(score, dtype=torch.float),
        }

def load_data_from_csv(file_path, n_augmentations=3):
    """
    Load dataset from CSV and augment the training data.
    """
    df = pd.read_csv(file_path)
    
    # Normalize scores between 0 and 1 if needed
    df['Score'] = df['Score'].apply(lambda x: sum(map(float, x.split(','))) / 2 / 100)  # Normalize score
    
    # Apply data augmentation
    augmented_rows = []
    for _, row in df.iterrows():
        original_text = row['Keywords']
        for _ in range(n_augmentations):
            augmented_text = augment_text(original_text)
            augmented_rows.append({'Keywords': augmented_text, 'Score': row['Score'], 'Class': row['Class']})
    
    # Append augmented rows to the original dataset
    augmented_df = pd.DataFrame(augmented_rows)
    df = pd.concat([df, augmented_df], ignore_index=True)

    return df

def load_dataset(file_path, batch_size=16, max_len=128, augment=True):
    """
    Loads the dataset from CSV, tokenizes the inputs, and returns DataLoader for training and validation.
    """
    # Load and optionally augment the data
    df = load_data_from_csv(file_path) if augment else pd.read_csv(file_path)
    data = df.to_dict(orient='records')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(data, tokenizer, max_len=max_len)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
