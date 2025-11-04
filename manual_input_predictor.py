"""
Manual Input Startup Success Predictor
Loads trained models, integrates semantic scoring, and predicts success
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List
from semantic_scorer import SemanticScorer

warnings.filterwarnings('ignore')


class ManualInputPredictor:
    """Predict startup success using manual input with semantic scoring"""

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

        # Initialize semantic scorer
        print("Loading semantic scorer...")
        self.semantic_scorer = SemanticScorer(batch_size=32)

        # Expected feature columns (from training)
        self.expected_features = [
            'funding_total_usd',
            'founder_education_quality',
            'founder_technical_background',
            'founder_business_background',
            'category_count',
            'cat_software',
            'cat_mobile',
            'cat_e-commerce',
            'cat_saas',
            'cat_health_care',
            'cat_finance',
            'cat_education',
            'cat_analytics',
            'cat_artificial_intelligence',
            'cat_enterprise_software',
            'cat_biotechnology',
            'cat_marketplace',
            'cat_social_media',
            'cat_fintech',
            'cat_gaming',
            'cat_advertising',
            'cat_cloud_computing',
            'cat_internet',
            'cat_apps',
            'cat_medical',
            'cat_real_estate',
            'cat_retail',
            'cat_social_network',
            'innovation_score',
            'confidence_score',
            'market_clarity_score',
            'overall_quality_score'
        ]

        print(f"✅ Model loaded: {type(self.model).__name__}")
        print(f"✅ Decision threshold: {threshold}")
        print(f"✅ Semantic scorer ready\n")

    def preprocess_features(self, startup_data: Dict) -> pd.DataFrame:
        """
        Preprocess features to match training format

        Args:
            startup_data: Dictionary with startup information

        Returns:
            DataFrame with processed features
        """
        # Start with default values
        features = {
            'funding_total_usd': 0,
            'founder_education_quality': 0,  # 0=Unknown, 1=Good, 2=Elite
            'founder_technical_background': 0,
            'founder_business_background': 0,
            'category_count': 0
        }

        # Initialize all category columns to 0
        category_columns = [
            'cat_software', 'cat_mobile', 'cat_e-commerce', 'cat_saas',
            'cat_health_care', 'cat_finance', 'cat_education', 'cat_analytics',
            'cat_artificial_intelligence', 'cat_enterprise_software',
            'cat_biotechnology', 'cat_marketplace', 'cat_social_media',
            'cat_fintech', 'cat_gaming', 'cat_advertising', 'cat_cloud_computing',
            'cat_internet', 'cat_apps', 'cat_medical', 'cat_real_estate',
            'cat_retail', 'cat_social_network'
        ]

        for cat in category_columns:
            features[cat] = 0

        # Update with provided values
        features.update(startup_data)

        # Calculate category_count from selected categories
        if 'categories' in startup_data:
            category_count = 0
            for cat in startup_data['categories']:
                cat_column = f'cat_{cat.lower().replace(" ", "_").replace("-", "_")}'
                if cat_column in features:
                    features[cat_column] = 1
                    category_count += 1
            features['category_count'] = category_count

        # Compute semantic scores if description provided
        if 'description' in startup_data and startup_data['description']:
            scores_df = self.semantic_scorer.score_descriptions([startup_data['description']])
            # Use default values if NaN
            features['innovation_score'] = scores_df['innovation_score'].iloc[0] if pd.notna(scores_df['innovation_score'].iloc[0]) else 50.0
            features['confidence_score'] = scores_df['confidence_score'].iloc[0] if pd.notna(scores_df['confidence_score'].iloc[0]) else 50.0
            features['market_clarity_score'] = scores_df['market_clarity_score'].iloc[0] if pd.notna(scores_df['market_clarity_score'].iloc[0]) else 50.0
        else:
            # Default scores if no description
            features['innovation_score'] = 50.0
            features['confidence_score'] = 50.0
            features['market_clarity_score'] = 50.0

        # Calculate overall quality score
        features['overall_quality_score'] = (
            features['innovation_score'] * 0.4 +
            features['confidence_score'] * 0.3 +
            features['market_clarity_score'] * 0.3
        )

        # Convert to DataFrame and ensure all expected features exist
        df = pd.DataFrame([features])

        # Add missing features with default value 0
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0

        # Return in correct order
        return df[self.expected_features]

    def predict_single(self, startup_data: Dict) -> Dict:
        """
        Predict success for a single startup

        Args:
            startup_data: Dictionary with startup features

        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        X = self.preprocess_features(startup_data)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        probability = self.model.predict_proba(X_scaled)[0, 1]
        prediction = int(probability >= self.threshold)

        # Compile result
        result = {
            'name': startup_data.get('name', 'Startup'),
            'prediction': 'SUCCESS' if prediction == 1 else 'FAILURE',
            'confidence': float(probability),
            'success_probability': float(probability * 100),
            'failure_probability': float((1 - probability) * 100),
            'risk_level': self._get_risk_level(probability),
            'description': startup_data.get('description', 'N/A')[:200],
            'funding_total_usd': startup_data.get('funding_total_usd', 0),
            'category_count': int(X['category_count'].iloc[0]),
            'innovation_score': float(round(X['innovation_score'].iloc[0], 2)),
            'confidence_score': float(round(X['confidence_score'].iloc[0], 2)),
            'market_clarity_score': float(round(X['market_clarity_score'].iloc[0], 2)),
            'overall_quality_score': float(round(X['overall_quality_score'].iloc[0], 2))
        }

        # Add insights
        result['insights'] = self._generate_insights(startup_data, X)
        result['recommendation'] = self._get_recommendation(probability)

        return result

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

    def _generate_insights(self, startup_data: Dict, features_df: pd.DataFrame) -> List[str]:
        """Generate insights based on features"""
        insights = []

        # Funding insights
        funding = startup_data.get('funding_total_usd', 0)
        if funding > 10000000:
            insights.append('✅ Strong funding (>$10M) - significant investor confidence')
        elif funding > 1000000:
            insights.append('✅ Good funding level (>$1M) - solid backing')
        elif funding > 0:
            insights.append('⚠️ Limited funding - may need additional capital')
        else:
            insights.append('⚠️ No funding information - bootstrapped or early stage')

        # Education insights
        edu_quality = startup_data.get('founder_education_quality', 0)
        if edu_quality == 2:
            insights.append('✅ Elite founder education - top tier background')
        elif edu_quality == 1:
            insights.append('✅ Good founder education')

        # Technical background
        if startup_data.get('founder_technical_background', False):
            insights.append('✅ Technical founder(s) - strong execution capability')

        # Business background
        if startup_data.get('founder_business_background', False):
            insights.append('✅ Business-savvy founder(s) - strong market understanding')

        # Category insights
        category_count = features_df['category_count'].iloc[0]
        if category_count == 0:
            insights.append('⚠️ No categories specified - unclear market positioning')
        elif category_count >= 3:
            insights.append('✅ Multi-category focus - diverse market opportunity')
        else:
            insights.append('✅ Focused category positioning')

        # Semantic score insights
        innovation = features_df['innovation_score'].iloc[0]
        if innovation >= 75:
            insights.append('✅ Highly innovative description - cutting-edge approach')
        elif innovation >= 60:
            insights.append('✅ Good innovation indicators')

        market_clarity = features_df['market_clarity_score'].iloc[0]
        if market_clarity >= 75:
            insights.append('✅ Excellent market clarity - well-defined value proposition')
        elif market_clarity < 50:
            insights.append('⚠️ Market positioning needs clarification')

        confidence_score = features_df['confidence_score'].iloc[0]
        if confidence_score >= 70:
            insights.append('✅ Strong confidence signals - established presence')

        return insights

    def _get_recommendation(self, probability: float) -> str:
        """Generate investment recommendation"""
        if probability >= 0.7:
            return 'STRONG BUY - High probability of success'
        elif probability >= 0.6:
            return 'BUY - Good probability of success'
        elif probability >= 0.5:
            return 'HOLD - Moderate probability, consider more data'
        elif probability >= 0.4:
            return 'CAUTION - Below average probability'
        else:
            return 'AVOID - Low probability of success'

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


# Helper function for quick testing
def quick_test():
    """Quick test of the predictor"""
    print("Testing Manual Input Predictor\n" + "="*70)

    # Sample startup data
    sample_startup = {
        'name': 'TechCorp AI',
        'description': 'We use cutting-edge artificial intelligence and machine learning to revolutionize healthcare diagnostics, helping doctors detect diseases earlier with proven 95% accuracy.',
        'funding_total_usd': 5000000,
        'founder_education_quality': 2,  # Elite
        'founder_technical_background': True,
        'founder_business_background': False,
        'categories': ['artificial_intelligence', 'health_care', 'saas']
    }

    # Note: You need model files first
    try:
        predictor = ManualInputPredictor('catboost_tuned.pkl', 'scaler.pkl')
        result = predictor.predict_single(sample_startup)

        print(f"\n{'='*70}")
        print(f"Startup: {result['name']}")
        print(f"{'='*70}")
        print(f"Prediction: {result['prediction']}")
        print(f"Success Probability: {result['success_probability']:.1f}%")
        print(f"Risk Level: {result['risk_level']}")
        print(f"\nSemantic Scores:")
        print(f"  Innovation: {result['innovation_score']}")
        print(f"  Confidence: {result['confidence_score']}")
        print(f"  Market Clarity: {result['market_clarity_score']}")
        print(f"  Overall Quality: {result['overall_quality_score']}")
        print(f"\nRecommendation: {result['recommendation']}")
        print(f"\nKey Insights:")
        for insight in result['insights']:
            print(f"  {insight}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have best_model.pkl and scaler.pkl in the current directory")


if __name__ == "__main__":
    quick_test()
