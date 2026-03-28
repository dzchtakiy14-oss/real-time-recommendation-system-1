import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_age_categories, num_locations, num_publishers, num_periods, num_authors, emb_dim = 128):
        super().__init__()
        # === Embeddings ===
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.age_emb = nn.Embedding(num_age_categories, emb_dim)
        self.location_emb = nn.Embedding(num_locations, emb_dim)
        self.publisher_emb = nn.Embedding(num_publishers, emb_dim)
        self.period_emb = nn.Embedding(num_periods, emb_dim)
        self.author_emb = nn.Embedding(num_authors, emb_dim)
        # === Context Linear === 
        self.linear_hour_cos = nn.Linear(1, emb_dim)
        self.linear_hour_sin = nn.Linear(1, emb_dim)
        self.linear_day_cos = nn.Linear(1, emb_dim)
        self.linear_day_sin = nn.Linear(1, emb_dim)
        self.linear_month_cos = nn.Linear(1, emb_dim)
        self.linear_month_sin = nn.Linear(1, emb_dim)

        # === User MLP === 
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(emb_dim * 2, emb_dim)
        )
        # === Item MLP ===
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(emb_dim * 2, emb_dim)
        )
        # === Context ===
        self.context_mlp = nn.Sequential(
            nn.Linear(emb_dim * 6, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        # === Integrate Context Vector + User Vector ===
        self.context_user_vec = nn.Sequential(
            nn.Linear(emb_dim * 2 , emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2 , emb_dim)
        )

    def compute_context(self, hour_cos, hour_sin, day_cos, day_sin, month_cos, month_sin):
        # === Compute Context Vector === 
        hour_cos = self.linear_hour_cos(hour_cos)
        hour_sin = self.linear_hour_sin(hour_sin)

        day_cos = self.linear_day_cos(day_cos)
        day_sin = self.linear_day_sin(day_sin)

        month_cos = self.linear_month_cos(month_cos)
        month_sin = self.linear_month_sin(month_sin)

        context_cat = torch.cat([hour_cos, hour_sin, day_cos, day_sin, month_cos, month_sin], dim=1)

        context_vec = self.context_mlp(context_cat)

        context_vec_norm = F.normalize(context_vec, p=2, dim=1)

        return context_vec_norm


    def user_tower(self, user_idx, age_category, location, context_vec = None):
        # === Compute User Vec ===
        user_emb = self.user_emb(user_idx)
        age_emb = self.age_emb (age_category)
        location_emb = self.location_emb(location)
        # Merge Embeddings 
        user_cat = torch.cat([user_emb, age_emb, location_emb], dim=1)
        # Compute User Vec
        user_vec = self.user_mlp(user_cat)
        # Normalize User Vector
        final_vec_norm = F.normalize(user_vec)

        # === Compute Context Vector ===
        if context_vec is not None:
            user_context_vec = self.context_user_vec(final_vec_norm, context_vec)
            final_vec_norm = F.normalize(user_context_vec, dim = 1)

        return final_vec_norm
        
    def item_tower(self, publisher, period, author):
        # === Compute Item Vector ===
        publisher_emb = self.publisher_emb(publisher)
        period_emb = self.period_emb(period)
        author_emb = self.author_emb(author)
        
        item_cat = torch.cat([publisher_emb, period_emb, author_emb], dim=1)

        item_vec = self.item_mlp(item_cat)

        # === Normalize Item Vector === 
        item_vec_normalized = F.normalize(item_vec)

        return item_vec_normalized
    
    def forward(self, book_idx, publisher, period, author, user_idx, age_category, location, context_vec = None, interacted_items_vec = None):
        user_vec = self.user_tower(user_idx, age_category, location, context_vec , interacted_items_vec)
        item_vec = self.item_tower(book_idx, publisher, period, author)

        return user_vec, item_vec