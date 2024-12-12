from transformers import BertModel
import torch.nn as nn

class BERTForQuantification(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(BERTForQuantification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Separate heads for each class (Reliability, Privacy, Security)
        self.reliability_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.privacy_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.security_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.resilience_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.safety_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, class_type):
        # BERT pooled output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Use the correct regression head based on class_type
        if class_type == "Reliability":
            score = self.reliability_head(pooled_output)
        elif class_type == "Privacy":
            score = self.privacy_head(pooled_output)
        elif class_type == "Security":
            score = self.security_head(pooled_output)
        elif class_type == "Resilience":
            score = self.resilience_head(pooled_output)
        elif class_type == "Safety":
            score = self.safety_head(pooled_output)
        else:
            raise ValueError(f"Invalid class_type. Must be one of: Reliability, Privacy, Security.")
        
        return score
