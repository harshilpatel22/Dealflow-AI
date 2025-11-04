"""
Flask Backend API for Startup Success Prediction
Manual input with semantic scoring integration
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from datetime import datetime
from manual_input_predictor import ManualInputPredictor

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Model paths
MODEL_PATH = 'catboost_tuned.pkl'
SCALER_PATH = 'scaler.pkl'
THRESHOLD = 0.45  # Optimized threshold

# Initialize predictor (will fail gracefully if models not found)
try:
    predictor = ManualInputPredictor(MODEL_PATH, SCALER_PATH, THRESHOLD)
    PREDICTOR_LOADED = True
except Exception as e:
    print(f"⚠️ Warning: Could not load predictor: {str(e)}")
    print("Upload your model files (best_model.pkl, scaler.pkl) to enable predictions")
    PREDICTOR_LOADED = False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'predictor_loaded': PREDICTOR_LOADED,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict success for startup data with semantic scoring

    Request body:
    {
        "name": "TechCorp AI",
        "description": "We use AI to revolutionize healthcare...",
        "funding_total_usd": 5000000,
        "founder_education_quality": 2,
        "founder_technical_background": true,
        "founder_business_background": false,
        "categories": ["artificial_intelligence", "health_care", "saas"]
    }
    """
    try:
        if not PREDICTOR_LOADED:
            return jsonify({'error': 'Predictor not loaded. Upload model files.'}), 503

        data = request.json
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # Validate required fields
        if 'description' not in data or not data['description']:
            return jsonify({'error': 'Description is required for semantic scoring'}), 400

        # Run prediction with semantic scoring
        result = predictor.predict_single(data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of available categories"""
    categories = [
        {'value': 'software', 'label': 'Software'},
        {'value': 'mobile', 'label': 'Mobile'},
        {'value': 'e-commerce', 'label': 'E-Commerce'},
        {'value': 'saas', 'label': 'SaaS'},
        {'value': 'health_care', 'label': 'Health Care'},
        {'value': 'finance', 'label': 'Finance'},
        {'value': 'education', 'label': 'Education'},
        {'value': 'analytics', 'label': 'Analytics'},
        {'value': 'artificial_intelligence', 'label': 'Artificial Intelligence'},
        {'value': 'enterprise_software', 'label': 'Enterprise Software'},
        {'value': 'biotechnology', 'label': 'Biotechnology'},
        {'value': 'marketplace', 'label': 'Marketplace'},
        {'value': 'social_media', 'label': 'Social Media'},
        {'value': 'fintech', 'label': 'FinTech'},
        {'value': 'gaming', 'label': 'Gaming'},
        {'value': 'advertising', 'label': 'Advertising'},
        {'value': 'cloud_computing', 'label': 'Cloud Computing'},
        {'value': 'internet', 'label': 'Internet'},
        {'value': 'apps', 'label': 'Apps'},
        {'value': 'medical', 'label': 'Medical'},
        {'value': 'real_estate', 'label': 'Real Estate'},
        {'value': 'retail', 'label': 'Retail'},
        {'value': 'social_network', 'label': 'Social Network'}
    ]
    return jsonify({'categories': categories})


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Get feature importance from model"""
    try:
        if not PREDICTOR_LOADED:
            return jsonify({'error': 'Predictor not loaded'}), 503

        importance_df = predictor.get_feature_importance(top_n=15)

        if importance_df.empty:
            return jsonify({'error': 'Feature importance not available for this model'}), 404

        return jsonify({
            'features': importance_df.to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🚀 STARTUP SUCCESS PREDICTION API - Manual Input Edition")
    print("="*70)
    print(f"Predictor loaded: {'✅ Yes' if PREDICTOR_LOADED else '❌ No (upload model files)'}")
    print(f"Semantic scoring: {'✅ Integrated' if PREDICTOR_LOADED else '❌ Not available'}")
    print(f"Server starting on http://localhost:5002")
    print("="*70)
    print("\nAvailable endpoints:")
    print("  GET  /                        - Dashboard UI")
    print("  GET  /api/status              - API status")
    print("  POST /api/predict             - Predict startup success")
    print("  GET  /api/categories          - Get available categories")
    print("  GET  /api/feature-importance  - Get model feature importance")
    print("="*70)

    app.run(debug=True, host='0.0.0.0', port=5002)
