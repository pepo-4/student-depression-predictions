"""
Inference utilities for the Depression Risk Assessment Pipeline.

Handles:
- Loading trained models (MCA, GMM, Logistic Regression)
- Transforming user input through MCA
- Cluster assignment via GMM
- Log-odds computation from statsmodels coefficients
- Probability prediction and risk assessment
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class InferencePipeline:
    """Complete inference pipeline for depression risk assessment."""
    
    def __init__(self, models_dir: str = './models'):
        """
        Initialize pipeline by loading all trained models.
        
        Args:
            models_dir: Directory path containing all pickled models
        """
        self.models_dir = Path(models_dir)
        self._load_models()
        
    def _load_models(self):
        """Load all models and artifacts from disk."""
        self.mca = joblib.load(self.models_dir / 'mca.pkl')
        self.gmm = joblib.load(self.models_dir / 'gmm.pkl')
        self.training_columns = joblib.load(self.models_dir / 'columns.pkl')
        
        # Load 3 logistic models
        self.models = {
            0: joblib.load(self.models_dir / 'model_0.pkl'),
            1: joblib.load(self.models_dir / 'model_1.pkl'),
            2: joblib.load(self.models_dir / 'model_2.pkl')
        }
        
        # Load results (contains thresholds, metrics)
        self.results = {
            0: joblib.load(self.models_dir / 'res_0.pkl'),
            1: joblib.load(self.models_dir / 'res_1.pkl'),
            2: joblib.load(self.models_dir / 'res_2.pkl')
        }
        
    def transform_input(self, user_input: dict) -> pd.DataFrame:
        """
        Transform user input to match training data format.
        
        Steps:
        1. Create DataFrame from dict
        2. Add missing columns as None
        3. Reorder to match training columns
        4. Convert all to category type
        
        Args:
            user_input: Dict with feature values
            
        Returns:
            pd.DataFrame with proper format and dtypes
        """
        # Create initial DataFrame
        df = pd.DataFrame([user_input])
        
        # Add missing columns
        for col in self.training_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reorder to match training
        df = df[self.training_columns]
        
        # Convert all to category
        for col in df.columns:
            df[col] = df[col].astype('category')
        
        return df
    
    def get_mca_coordinates(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform input through MCA (trained on training data).
        
        Args:
            df: Properly formatted input DataFrame
            
        Returns:
            np.ndarray of shape (1, 2) with MCA coordinates
        """
        coords = self.mca.transform(df)
        # Ensure it's 2D and take only first 2 components if more exist
        if isinstance(coords, pd.DataFrame):
            coords = coords.iloc[:, :2].values
        elif isinstance(coords, np.ndarray):
            coords = coords[:, :2]
        return coords
    
    def get_cluster(self, mca_coords: np.ndarray) -> int:
        """
        Assign cluster using trained GMM.
        
        Args:
            mca_coords: MCA coordinates from transform
            
        Returns:
            int: Cluster ID (0, 1, or 2)
        """
        cluster = self.gmm.predict(mca_coords)[0]
        return int(cluster)
    
    def compute_log_odds(self, user_input: dict, cluster: int) -> float:
        """
        Manually compute log-odds from statsmodels coefficients.
        
        Logic:
        1. Get model for cluster
        2. Start with intercept (const)
        3. For each feature in user_input:
           - Create dummy name: "Feature_Value"
           - Find matching coefficient
           - Add to log_odds if feature is active
        
        Args:
            user_input: Original input dict
            cluster: Cluster ID
            
        Returns:
            float: log-odds value
        """
        model = self.models[cluster]
        params = model.params
        
        # Start with intercept
        log_odds = float(params.get('const', 0.0))
        
        # Collect active features for later reporting
        active_features = []
        
        # Process each feature value
        for feat_name, user_val in user_input.items():
            if user_val is None:
                continue
            
            # Create dummy name: clean the value string
            clean_val = str(user_val).replace(' ', '_').replace('/', '_')
            dummy_name = f"{feat_name}_{clean_val}"
            
            # Check if this dummy exists in model coefficients
            if dummy_name in params.index:
                coef = float(params[dummy_name])
                log_odds += coef
                active_features.append((dummy_name, coef))
        
        return log_odds, active_features
    
    def predict_probability(self, log_odds: float) -> float:
        """
        Convert log-odds to probability using sigmoid.
        
        P = 1 / (1 + exp(-log_odds))
        
        Args:
            log_odds: Log-odds value
            
        Returns:
            float: Probability in [0, 1]
        """
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        return probability
    
    def get_threshold(self, cluster: int) -> float:
        """
        Get decision threshold for cluster (train prevalence).
        
        Args:
            cluster: Cluster ID
            
        Returns:
            float: Threshold value
        """
        return float(self.results[cluster].get('Threshold', 0.5))
    
    def predict_pipeline(self, user_input: dict) -> dict:
        """
        Complete inference pipeline.
        
        Args:
            user_input: Dict with user responses
            
        Returns:
            dict with keys:
                - cluster: int (0, 1, 2)
                - probability: float
                - threshold: float
                - risk: bool (True if prob >= threshold)
                - positive_factors: list of (name, coef) tuples
                - negative_factors: list of (name, coef) tuples
                - message: str (risk description)
        """
        # Step 1: Transform input
        df = self.transform_input(user_input)
        
        # Step 2: Get MCA coordinates
        mca_coords = self.get_mca_coordinates(df)
        
        # Step 3: Get cluster
        cluster = self.get_cluster(mca_coords)
        
        # Step 4: Compute log-odds and get active features
        log_odds, active_features = self.compute_log_odds(user_input, cluster)
        
        # Step 5: Get probability
        probability = self.predict_probability(log_odds)
        
        # Step 6: Get threshold
        threshold = self.get_threshold(cluster)
        
        # Step 7: Determine risk
        is_risk = probability >= threshold
        
        # Step 8: Separate positive and negative factors
        positive_factors = [(f, c) for f, c in active_features if c > 0]
        negative_factors = [(f, c) for f, c in active_features if c < 0]
        
        # Sort by magnitude
        positive_factors.sort(key=lambda x: x[1], reverse=True)
        negative_factors.sort(key=lambda x: x[1])
        
        # Create message
        if is_risk:
            message = f"⚠️ RISCHIO RILEVATO (Prob: {probability:.1%} > Soglia: {threshold:.1%})"
        else:
            message = f"✅ SITUAZIONE SOTTO CONTROLLO (Prob: {probability:.1%} < Soglia: {threshold:.1%})"
        
        return {
            'cluster': cluster,
            'cluster_name': self._get_cluster_name(cluster),
            'probability': probability,
            'threshold': threshold,
            'is_risk': is_risk,
            'message': message,
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'log_odds': log_odds
        }
    
    @staticmethod
    def _get_cluster_name(cluster: int) -> str:
        """Map cluster ID to descriptive name."""
        names = {
            0: "Cluster 0 (HS Giovani)",
            1: "Cluster 1 (Uni Grandi)",
            2: "Cluster 2 (Uni Giovani)"
        }
        return names.get(cluster, "Sconosciuto")


def load_pipeline(models_dir: str = './models') -> InferencePipeline:
    """Convenience function to load the full pipeline."""
    return InferencePipeline(models_dir=models_dir)
