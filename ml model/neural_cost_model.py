"""
Neural Cost Model - ML Integration Component
Deep learning model that learns query execution costs from historical data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle


@dataclass
class QueryPlan:
    """Represents a query execution plan"""
    # Operator types in the plan
    operators: List[str]  # ['SeqScan', 'IndexScan', 'HashJoin', 'VectorSearch', ...]
    
    # Estimated cardinalities at each stage
    cardinalities: List[int]
    
    # Table sizes
    table_sizes: List[int]
    
    # Selectivities
    selectivities: List[float]
    
    # Index usage
    uses_indexes: List[bool]
    
    # Vector search parameters
    has_vector_search: bool
    vector_k: int
    vector_table_size: int
    
    # Join information
    num_joins: int
    join_methods: List[str]  # ['hash', 'nested_loop', 'merge']
    
    # Actual execution cost (for training)
    actual_cost: Optional[float] = None  # milliseconds
    actual_rows: Optional[int] = None


class QueryPlanDataset(Dataset):
    """PyTorch dataset for query plans"""
    
    def __init__(self, plans: List[QueryPlan], max_operators: int = 20):
        self.plans = plans
        self.max_operators = max_operators
        
        # Build vocabulary for operators
        self.operator_vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary mapping operators to indices"""
        all_operators = set()
        for plan in self.plans:
            all_operators.update(plan.operators)
            all_operators.update(plan.join_methods)
        
        vocab = {op: idx for idx, op in enumerate(sorted(all_operators))}
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab)
        
        return vocab
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        plan = self.plans[idx]
        
        # Encode plan as features
        features = self._encode_plan(plan)
        
        # Target is log(cost) for better training
        target = np.log10(max(plan.actual_cost, 0.1)) if plan.actual_cost else 0.0
        
        return (
            torch.FloatTensor(features),
            torch.FloatTensor([target])
        )
    
    def _encode_plan(self, plan: QueryPlan) -> np.ndarray:
        """Encode plan as fixed-size feature vector"""
        features = []
        
        # Operator sequence (one-hot encoded and padded)
        op_encoding = np.zeros(self.max_operators)
        for i, op in enumerate(plan.operators[:self.max_operators]):
            op_idx = self.operator_vocab.get(op, self.operator_vocab['<UNK>'])
            op_encoding[i] = op_idx
        features.extend(op_encoding)
        
        # Cardinality features (log-scale, padded)
        card_features = np.zeros(self.max_operators)
        for i, card in enumerate(plan.cardinalities[:self.max_operators]):
            card_features[i] = np.log10(max(card, 1))
        features.extend(card_features)
        
        # Selectivity features (padded)
        sel_features = np.zeros(self.max_operators)
        for i, sel in enumerate(plan.selectivities[:self.max_operators]):
            sel_features[i] = sel
        features.extend(sel_features)
        
        # Global plan statistics
        features.extend([
            np.log10(max(sum(plan.table_sizes), 1)),  # Total data size
            np.mean(plan.selectivities) if plan.selectivities else 0.5,
            float(plan.has_vector_search),
            np.log10(max(plan.vector_k, 1)) if plan.has_vector_search else 0,
            plan.num_joins,
            len(plan.operators),
            np.sum(plan.uses_indexes) / max(len(plan.uses_indexes), 1),
        ])
        
        return np.array(features, dtype=np.float32)


class PlanEncoderRNN(nn.Module):
    """
    Recurrent neural network for encoding query plans
    Uses LSTM to process sequential plan operators
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, operator_indices):
        """
        Args:
            operator_indices: (batch, max_ops)
        Returns:
            Final hidden state: (batch, hidden_dim)
        """
        # Embed operators
        embedded = self.embedding(operator_indices)  # (batch, max_ops, embed_dim)
        
        # Process with LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use final hidden state
        final_hidden = hidden[-1]  # (batch, hidden_dim)
        
        return final_hidden


class NeuralCostModel(nn.Module):
    """
    Neural network for predicting query execution cost
    Combines plan structure (via RNN) with tabular features
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_tabular_features: int = 50,
        embedding_dim: int = 64,
        lstm_hidden: int = 128,
        mlp_hidden: List[int] = [256, 128, 64]
    ):
        super().__init__()
        
        # Plan encoder (RNN)
        self.plan_encoder = PlanEncoderRNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=lstm_hidden,
            num_layers=2
        )
        
        # MLP for tabular features
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, mlp_hidden[0]),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden[0]),
            nn.Dropout(0.3),
            
            nn.Linear(mlp_hidden[0], mlp_hidden[1]),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden[1]),
            nn.Dropout(0.2),
        )
        
        # Combined prediction head
        combined_dim = lstm_hidden + mlp_hidden[1]
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden[2], 1)
        )
    
    def forward(self, operator_indices, tabular_features):
        """
        Args:
            operator_indices: (batch, max_ops) - operator sequence
            tabular_features: (batch, num_features) - cardinalities, selectivities, etc.
        
        Returns:
            Predicted log(cost): (batch, 1)
        """
        # Encode plan structure
        plan_embedding = self.plan_encoder(operator_indices)  # (batch, lstm_hidden)
        
        # Process tabular features
        tabular_embedding = self.tabular_mlp(tabular_features)  # (batch, mlp_hidden[1])
        
        # Concatenate both representations
        combined = torch.cat([plan_embedding, tabular_embedding], dim=1)
        
        # Predict cost
        log_cost = self.prediction_head(combined)
        
        return log_cost


class NeuralCostModelTrainer:
    """
    Training and inference for neural cost models
    """
    
    def __init__(
        self,
        vocab_size: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = NeuralCostModel(vocab_size=vocab_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.loss_fn = nn.MSELoss()
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_q_error': [],
            'val_q_error': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        q_errors = []
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Split features
            # Assuming features are concatenated: [ops, cards, sels, tabular]
            # This is simplified - adjust based on actual encoding
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_features[:, :20].long(), 
                                   batch_features[:, 20:])
            
            # Compute loss
            loss = self.loss_fn(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute Q-error
            pred_cost = 10 ** predictions.detach().cpu().numpy()
            true_cost = 10 ** batch_targets.detach().cpu().numpy()
            batch_q_errors = np.maximum(
                pred_cost / (true_cost + 1e-6),
                true_cost / (pred_cost + 1e-6)
            )
            q_errors.extend(batch_q_errors.flatten())
        
        avg_loss = total_loss / len(train_loader)
        median_q_error = np.median(q_errors)
        
        return {
            'loss': avg_loss,
            'median_q_error': median_q_error,
            '95th_q_error': np.percentile(q_errors, 95)
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data"""
        self.model.eval()
        total_loss = 0.0
        q_errors = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_features[:, :20].long(),
                                       batch_features[:, 20:])
                
                loss = self.loss_fn(predictions, batch_targets)
                total_loss += loss.item()
                
                # Q-error
                pred_cost = 10 ** predictions.cpu().numpy()
                true_cost = 10 ** batch_targets.cpu().numpy()
                batch_q_errors = np.maximum(
                    pred_cost / (true_cost + 1e-6),
                    true_cost / (pred_cost + 1e-6)
                )
                q_errors.extend(batch_q_errors.flatten())
        
        avg_loss = total_loss / len(val_loader)
        median_q_error = np.median(q_errors)
        
        return {
            'loss': avg_loss,
            'median_q_error': median_q_error,
            '95th_q_error': np.percentile(q_errors, 95)
        }
    
    def train(
        self,
        train_dataset: QueryPlanDataset,
        val_dataset: QueryPlanDataset,
        num_epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Record history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_q_error'].append(train_metrics['median_q_error'])
            self.training_history['val_q_error'].append(val_metrics['median_q_error'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                      f"Q-error: {train_metrics['median_q_error']:.2f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Q-error: {val_metrics['median_q_error']:.2f}")
        
        return self.training_history
    
    def predict_cost(self, plan: QueryPlan, dataset: QueryPlanDataset) -> float:
        """
        Predict execution cost for a query plan
        
        Returns:
            Estimated cost in milliseconds
        """
        self.model.eval()
        
        # Encode plan
        features = dataset._encode_plan(plan)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_cost = self.model(
                features_tensor[:, :20].long(),
                features_tensor[:, 20:]
            )
        
        cost = 10 ** log_cost.item()
        return max(cost, 0.1)  # At least 0.1ms
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
