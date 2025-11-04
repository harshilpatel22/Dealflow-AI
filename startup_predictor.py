"""
Startup Success Predictor
Loads trained models and predicts success for new startups
Provides confidence scores and feature importance
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class StartupSuccessPredictor:
    """Predict startup success using trained models"""

    def __init__(self, model_path: str = 'best_model.pkl',
                 scaler_path: str = 'scaler.pkl',
                 threshold: float = 0.45):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model pickle file
            scaler_path: Path to scaler pickle file
            threshold: Decision threshold (default 0.45 for better recall)
        """
        print("Loading models...")

        # Load model and scaler
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.threshold = threshold

        # Expected feature columns (from training)
        self.expected_features = [
            'funding_total_usd', 'status', 'city', 'website_accessible',
            'website_status_code', 'founder_count', 'founder_education_quality',
            'founder_technical_background', 'founder_business_background',
            'business_model_clarity', 'technology_stack', 'technology_count',
            'content_length', 'word_count', 'detected_language',
            'customer_mentions_count', 'revenue_mentions_count',
            'growth_mentions_count', 'team_size_count',
            'partnership_mentions', 'award_mentions', 'category_count'
        ]

        print(f"✅ Model loaded: {type(self.model).__name__}")
        print(f"✅ Decision threshold: {threshold}")

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features to match training format"""
        df_processed = df.copy()

        # Add default values for missing features
        default_values = {
            'funding_total_usd': 0,
            'status': 'operating',
            'city': 'Unknown',
            'website_status_code': 200,
            'customer_mentions_count': 0,
            'revenue_mentions_count': 0,
            'growth_mentions_count': 0,
            'team_size_count': 0,
            'partnership_mentions': 0,
            'award_mentions': 0,
            'category_count': 1
        }

        for feature, default in default_values.items():
            if feature not in df_processed.columns:
                df_processed[feature] = default

        # Encode categorical features
        categorical_mapping = {
            'founder_education_quality': {'Unknown': 0, 'Good': 1, 'Elite': 2},
            'business_model_clarity': {'Unknown': 0, 'B2C': 1, 'B2B': 2, 'SAAS': 3,
                                      'MARKETPLACE': 4, 'E-COMMERCE': 5},
            'detected_language': {'en': 0, 'es': 1, 'fr': 2, 'de': 3, 'other': 4}
        }

        for col, mapping in categorical_mapping.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map(
                    lambda x: mapping.get(x, 0) if pd.notna(x) else 0
                )

        # Convert boolean columns
        bool_cols = ['website_accessible', 'founder_technical_background',
                     'founder_business_background']
        for col in bool_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)

        # Encode remaining categoricals
        for col in df_processed.select_dtypes(include=['object']).columns:
            if col not in ['name', 'company_description', 'homepage_url']:
                try:
                    df_processed[col] = LabelEncoder().fit_transform(df_processed[col].astype(str))
                except:
                    df_processed[col] = 0

        # Select only expected features
        available_features = [f for f in self.expected_features if f in df_processed.columns]
        missing_features = [f for f in self.expected_features if f not in df_processed.columns]

        # Add missing features with default value 0
        for feature in missing_features:
            df_processed[feature] = 0

        # Return in correct order
        return df_processed[self.expected_features]

    def predict_single(self, startup_data: Dict) -> Dict:
        """
        Predict success for a single startup

        Args:
            startup_data: Dictionary with startup features

        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([startup_data])
        results = self.predict_batch(df)
        return results[0]

    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        """
        Predict success for multiple startups

        Args:
            df: DataFrame with startup features

        Returns:
            List of prediction dictionaries
        """
        # Store original data
        original_data = df.copy()

        # Preprocess
        X = self.preprocess_features(df)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        # Compile results
        results = []
        for i, (idx, row) in enumerate(original_data.iterrows()):
            result = {
                'name': row.get('name', f'Startup_{i+1}'),
                'url': row.get('homepage_url', 'N/A'),
                'prediction': 'SUCCESS' if predictions[i] == 1 else 'FAILURE',
                'confidence': float(probabilities[i]),
                'success_probability': float(probabilities[i] * 100),
                'failure_probability': float((1 - probabilities[i]) * 100),
                'risk_level': self._get_risk_level(probabilities[i]),
                'description': row.get('company_description', 'N/A')[:200],
                'founder_count': int(row.get('founder_count', 0)),
                'technology_count': int(row.get('technology_count', 0)),
                'business_model': row.get('business_model_clarity', 'Unknown')
            }
            results.append(result)

        return results

    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on success probability"""
        if probability >= 0.7:
            return 'LOW'
        elif probability >= 0.5:
            return 'MEDIUM'
        elif probability >= 0.3:
            return 'HIGH'
        else:
            return 'VERY HIGH'

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_

                importance_df = pd.DataFrame({
                    'feature': self.expected_features,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)

                return importance_df
        except:
            pass

        return pd.DataFrame()

    def analyze_startup(self, startup_data: Dict) -> Dict:
        """
        Comprehensive analysis of a single startup

        Returns detailed insights and recommendations
        """
        prediction = self.predict_single(startup_data)

        # Add insights
        insights = []

        # Founder analysis
        founder_count = startup_data.get('founder_count', 0)
        if founder_count == 0:
            insights.append('⚠️ No founder information available - increases risk')
        elif founder_count >= 2:
            insights.append('✅ Multiple founders - good sign for success')

        # Technical background
        if startup_data.get('founder_technical_background'):
            insights.append('✅ Technical founder(s) detected')

        # Education
        edu_quality = startup_data.get('founder_education_quality', 'Unknown')
        if edu_quality == 'Elite':
            insights.append('✅ Elite education background detected')

        # Technology
        tech_count = startup_data.get('technology_count', 0)
        if tech_count >= 2:
            insights.append(f'✅ Uses {tech_count} technologies - tech-forward')

        # Business model
        biz_model = startup_data.get('business_model_clarity', 'Unknown')
        if biz_model != 'Unknown':
            insights.append(f'✅ Clear business model: {biz_model}')
        else:
            insights.append('⚠️ Business model unclear')

        # Website
        if not startup_data.get('website_accessible', False):
            insights.append('❌ Website not accessible - major red flag')

        prediction['insights'] = insights
        prediction['recommendation'] = self._get_recommendation(prediction)

        return prediction

    def _get_recommendation(self, prediction: Dict) -> str:
        """Generate investment recommendation"""
        confidence = prediction['confidence']

        if confidence >= 0.7:
            return 'STRONG BUY - High probability of success'
        elif confidence >= 0.5:
            return 'BUY - Moderate probability of success'
        elif confidence >= 0.3:
            return 'HOLD - Uncertain, needs more data'
        else:
            return 'AVOID - Low probability of success'


# Helper functions
def predict_from_csv(csv_path: str,
                     model_path: str = 'best_model.pkl',
                     output_path: str = 'predictions.csv') -> pd.DataFrame:
    """
    Predict success for startups in CSV file

    Args:
        csv_path: Path to CSV with startup data
        model_path: Path to trained model
        output_path: Path to save predictions

    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Initialize predictor
    predictor = StartupSuccessPredictor(model_path)

    # Predict
    results = predictor.predict_batch(df)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

    return results_df


if __name__ == "__main__":
    # Example usage
    print("Startup Success Predictor\n" + "="*50)

    # Sample startup data
    sample_startup = {
        'name': 'TechCorp AI',
        'homepage_url': 'https://techcorp.ai',
        'company_description': 'AI-powered platform for enterprise automation',
        'founder_count': 2,
        'founder_education_quality': 'Elite',
        'founder_technical_background': True,
        'founder_business_background': False,
        'business_model_clarity': 'SAAS',
        'technology_count': 3,
        'website_accessible': True,
        'content_length': 5000,
        'word_count': 800
    }

    # Note: You need to have the model files first
    # predictor = StartupSuccessPredictor('best_model.pkl')
    # result = predictor.analyze_startup(sample_startup)
    # print(f"\nPrediction: {result['prediction']}")
    # print(f"Confidence: {result['success_probability']:.1f}%")
    # print(f"Risk Level: {result['risk_level']}")
