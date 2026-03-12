"""
ML-based Selectivity Estimator - Cost Model Researcher Component
Uses machine learning to predict query selectivity more accurately than histograms
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QueryFeatures:
    """Features extracted from a query for ML prediction"""
    # Table statistics
    table_row_count: int
    table_size_mb: float
    
    # Filter predicates
    num_equality_predicates: int
    num_range_predicates: int
    num_like_predicates: int
    
    # Predicate selectivity hints (from histograms)
    min_histogram_selectivity: float
    max_histogram_selectivity: float
    avg_histogram_selectivity: float
    
    # Column statistics
    num_distinct_values: List[int]
    null_fraction: List[float]
    
    # Index availability
    has_index: List[bool]
    
    # Vector search features
    has_vector_search: bool
    vector_k: int
    vector_table_ratio: float  # k / table_size
    
    # Join features
    num_joins: int
    join_selectivity_estimates: List[float]
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        features = [
            np.log10(max(self.table_row_count, 1)),
            np.log10(max(self.table_size_mb, 0.1)),
            self.num_equality_predicates,
            self.num_range_predicates,
            self.num_like_predicates,
            self.min_histogram_selectivity,
            self.max_histogram_selectivity,
            self.avg_histogram_selectivity,
            float(self.has_vector_search),
            np.log10(max(self.vector_k, 1)) if self.has_vector_search else 0,
            self.vector_table_ratio,
            self.num_joins,
        ]
        
        # Aggregate stats for multi-column predicates
        if self.num_distinct_values:
            features.extend([
                np.mean(np.log10(np.maximum(self.num_distinct_values, 1))),
                np.std(np.log10(np.maximum(self.num_distinct_values, 1))),
            ])
        else:
            features.extend([0, 0])
        
        if self.null_fraction:
            features.extend([
                np.mean(self.null_fraction),
                np.max(self.null_fraction),
            ])
        else:
            features.extend([0, 0])
        
        if self.has_index:
            features.append(np.mean([1.0 if x else 0.0 for x in self.has_index]))
        else:
            features.append(0.0)
        
        if self.join_selectivity_estimates:
            features.extend([
                np.mean(self.join_selectivity_estimates),
                np.min(self.join_selectivity_estimates),
            ])
        else:
            features.extend([0.5, 0.5])
        
        return np.array(features, dtype=np.float32)


@dataclass
class TrainingExample:
    """A training example with features and actual selectivity"""
    features: QueryFeatures
    actual_selectivity: float
    actual_cardinality: int
    query_id: str


class MLSelectivityEstimator:
    """
    Machine learning-based selectivity estimator
    Learns from query execution history to improve predictions
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest', 'gradient_boost', or 'neural_net'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_examples = []
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        elif self.model_type == 'neural_net':
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def add_training_example(
        self,
        features: QueryFeatures,
        actual_selectivity: float,
        actual_cardinality: int,
        query_id: str
    ):
        """Add a training example from query execution"""
        example = TrainingExample(
            features=features,
            actual_selectivity=actual_selectivity,
            actual_cardinality=actual_cardinality,
            query_id=query_id
        )
        self.training_examples.append(example)
    
    def train(self, min_examples: int = 50) -> Dict[str, float]:
        """
        Train the model on collected examples
        
        Returns:
            Training metrics (R^2, MAE, etc.)
        """
        if len(self.training_examples) < min_examples:
            return {
                'status': 'insufficient_data',
                'n_examples': len(self.training_examples),
                'min_required': min_examples
            }
        
        # Prepare training data
        X = np.array([ex.features.to_feature_vector() 
                     for ex in self.training_examples])
        y = np.array([ex.actual_selectivity 
                     for ex in self.training_examples])
        
        # Handle log transform for selectivity (avoid log(0))
        y_log = np.log10(np.maximum(y, 1e-6))
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y_log)
        self.is_trained = True
        
        # Evaluate on training data (for monitoring)
        y_pred_log = self.model.predict(X_scaled)
        y_pred = 10 ** y_pred_log
        
        # Compute metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Q-error: max(pred/actual, actual/pred)
        q_errors = np.maximum(y_pred / (y + 1e-6), y / (y_pred + 1e-6))
        median_q_error = np.median(q_errors)
        percentile_95_q_error = np.percentile(q_errors, 95)
        
        return {
            'status': 'trained',
            'n_examples': len(self.training_examples),
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'median_q_error': median_q_error,
            '95th_percentile_q_error': percentile_95_q_error
        }
    
    def predict_selectivity(
        self,
        features: QueryFeatures,
        fallback_selectivity: Optional[float] = None
    ) -> float:
        """
        Predict selectivity for a query
        
        Args:
            features: Query features
            fallback_selectivity: Use this if model not trained
            
        Returns:
            Predicted selectivity (0.0 to 1.0)
        """
        if not self.is_trained:
            if fallback_selectivity is not None:
                return fallback_selectivity
            # Default heuristic
            return features.avg_histogram_selectivity
        
        # Extract and scale features
        X = features.to_feature_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict (in log space)
        y_pred_log = self.model.predict(X_scaled)[0]
        selectivity = 10 ** y_pred_log
        
        # Clamp to valid range
        selectivity = np.clip(selectivity, 0.0, 1.0)
        
        return selectivity
    
    def predict_cardinality(
        self,
        features: QueryFeatures,
        table_size: int,
        fallback_selectivity: Optional[float] = None
    ) -> int:
        """Predict output cardinality"""
        selectivity = self.predict_selectivity(features, fallback_selectivity)
        cardinality = int(table_size * selectivity)
        return max(1, cardinality)  # At least 1 row
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'model_type': self.model_type,
            'n_examples': len(self.training_examples)
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.model_type = model_data['model_type']
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances (for Random Forest and Gradient Boosting)"""
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            feature_names = [
                'log_table_rows', 'log_table_size_mb',
                'num_eq_predicates', 'num_range_predicates', 'num_like_predicates',
                'min_hist_sel', 'max_hist_sel', 'avg_hist_sel',
                'has_vector_search', 'log_vector_k', 'vector_table_ratio',
                'num_joins',
                'avg_log_ndv', 'std_log_ndv',
                'avg_null_frac', 'max_null_frac',
                'avg_has_index',
                'avg_join_sel', 'min_join_sel'
            ]
            
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
        
        return {}
    
    def explain_prediction(
        self,
        features: QueryFeatures,
        top_k: int = 5
    ) -> str:
        """Explain why the model made a particular prediction"""
        if not self.is_trained:
            return "Model not trained yet"
        
        selectivity = self.predict_selectivity(features)
        
        explanation = [
            f"Predicted Selectivity: {selectivity:.4f}",
            f"Model Type: {self.model_type}",
            "",
            "Top Contributing Features:"
        ]
        
        # Get feature importances
        importances = self.get_feature_importance()
        if importances:
            sorted_features = sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            for feat_name, importance in sorted_features:
                explanation.append(f"  {feat_name}: {importance:.4f}")
        
        explanation.extend([
            "",
            "Query Features:",
            f"  Table Size: {features.table_row_count:,} rows",
            f"  Equality Predicates: {features.num_equality_predicates}",
            f"  Range Predicates: {features.num_range_predicates}",
            f"  Histogram Selectivity: {features.avg_histogram_selectivity:.4f}",
            f"  Vector Search: {features.has_vector_search} (k={features.vector_k})",
            f"  Joins: {features.num_joins}",
        ])
        
        return "\n".join(explanation)


class EnsembleSelectivityEstimator:
    """
    Ensemble of multiple estimators for robust predictions
    Combines histogram-based, ML-based, and sampling-based estimates
    """
    
    def __init__(self):
        self.ml_estimator = MLSelectivityEstimator(model_type='random_forest')
        self.weights = {
            'histogram': 0.3,
            'ml': 0.5,
            'sampling': 0.2
        }
    
    def predict_selectivity(
        self,
        features: QueryFeatures,
        histogram_selectivity: float,
        sampling_selectivity: Optional[float] = None
    ) -> float:
        """
        Combine multiple estimates with learned weights
        
        Args:
            features: Query features for ML prediction
            histogram_selectivity: Traditional histogram estimate
            sampling_selectivity: Estimate from sampling (if available)
            
        Returns:
            Weighted ensemble prediction
        """
        estimates = {}
        
        # Histogram estimate
        estimates['histogram'] = histogram_selectivity
        
        # ML estimate (if trained)
        if self.ml_estimator.is_trained:
            estimates['ml'] = self.ml_estimator.predict_selectivity(
                features,
                fallback_selectivity=histogram_selectivity
            )
        else:
            # Fall back to histogram
            estimates['ml'] = histogram_selectivity
        
        # Sampling estimate (if available)
        if sampling_selectivity is not None:
            estimates['sampling'] = sampling_selectivity
        else:
            estimates['sampling'] = histogram_selectivity
        
        # Weighted average
        total_weight = sum(self.weights.values())
        weighted_sum = sum(
            self.weights[name] * value
            for name, value in estimates.items()
        )
        
        final_estimate = weighted_sum / total_weight
        
        return np.clip(final_estimate, 0.0, 1.0)
    
    def update_weights_from_feedback(
        self,
        errors: Dict[str, List[float]]
    ):
        """
        Update ensemble weights based on prediction errors
        
        Args:
            errors: Dict mapping estimator names to lists of Q-errors
        """
        # Compute inverse of median error as weight
        for name in self.weights.keys():
            if name in errors and errors[name]:
                median_error = np.median(errors[name])
                # Lower error -> higher weight
                self.weights[name] = 1.0 / max(median_error, 0.1)
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
