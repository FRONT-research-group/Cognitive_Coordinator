import torch
from transformers import BertTokenizer
from src.models.models import BERTForQuantification
from src.utils.utils import load_model
from typing import List, Dict
from collections import defaultdict
from src.utils.utils import inference, compute_wtf, calibrate

class ClassificationComponentWrapper():

    def __init__(self):
        super(ClassificationComponentWrapper, self).__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model('src/models/final_bert_model.pth', self.device)

    def caclulate_nlotw(self, input_data) -> Dict[str, List[float]]:
        """Groups input data by label and infers a model for each text in the label."""
        grouped_data = defaultdict(list)
        
        # Group texts by their labels
        for item in input_data:
            grouped_data[item.label].append(item.text)
        
        # Perform model inference
        scores_per_label = {}
        for label, texts in grouped_data.items():
            scores_per_label[label] = [inference(text, label, self.model, self.tokenizer, self.device) for text in texts]
        
        # Compute and normalize W_TF
        wtf = compute_wtf(scores_per_label)
        
        # Compute final nLoTW using normalized W_TF * average(REG_TFj)
        nlotw = {label: wtf[label] * (sum(scores) / len(scores)) for label, scores in scores_per_label.items()}
        
        # Normalize final nlotw values to ensure sum(nlotw) == 1
        total_nlotw = sum(nlotw.values())
        if total_nlotw > 0:
            nlotw = {label: 100 * score / total_nlotw for label, score in nlotw.items()}  # Normalize final scores
        return nlotw
    
    def calculate_clotw(self, input_data) -> Dict[str, float]:
        """Groups input data by label, performs inference, and calculates normalized nLoTW."""
        grouped_data = defaultdict(list)
        
        # Group texts by their labels
        for item in input_data:
            grouped_data[item.label].append(item.text)
        
        # Perform model inference
        scores_per_label = {}
        for label, texts in grouped_data.items():
            scores_per_label[label] = [inference(text, label, self.model, self.tokenizer, self.device) for text in texts]
        
        cscores_per_label = calibrate(scores_per_label)

        print(f"Calibrated scores: {cscores_per_label}")

        # Compute and normalize W_TF
        wtf = compute_wtf(scores_per_label)
        
        # Compute final nLoTW using normalized W_TF * average(REG_TFj)
        clotw = {label: wtf[label] * calibrated_score for label, calibrated_score in cscores_per_label}

        
        # Normalize final cLoTw values to ensure sum(cLoTw) == 1
        total_clotw = sum(clotw.values())
        if total_clotw > 0:
            clotw = {label: 100 * score / total_clotw for label, score in clotw.items()}  # Normalize final scores
        
        return clotw
