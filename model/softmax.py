import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxLossWithCorrection(nn.Module):
    """
    Softmax Loss مع تصحيح تردد العينات (LogQ Correction)
    - تحل مشكلة تحيز العناصر الشائعة (popularity bias)
    - تستخدم في أنظمة الإنتاج الكبرى
    المرجع: Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations [citation:7]
    """
    def __init__(self, temperature=0.05, use_correction=True):
        super().__init__()
        self.temperature = temperature
        self.use_correction = use_correction
        
    def forward(self, user_embeddings, item_embeddings, item_freq=None):
        """
        Args:
            user_embeddings: [batch_size, embed_dim] - متجهات المستخدمين
            item_embeddings: [batch_size, embed_dim] - متجهات العناصر الموجبة
            item_freq: [vocab_size] أو [batch_size] - تواتر ظهور كل عنصر
                      (مطلوب إذا use_correction=True)
        """
        batch_size = user_embeddings.size(0)
        

        logits = torch.matmul(user_embeddings, item_embeddings.T) 

        logits = logits / self.temperature

        if self.use_correction and item_freq is not None:
            log_q = torch.log(item_freq + 1e-10)  
            logits = logits - log_q.unsqueeze(0)  
        
        targets = torch.arange(batch_size, device=user_embeddings.device)
        
        loss = F.cross_entropy(logits, targets)
        
        return loss