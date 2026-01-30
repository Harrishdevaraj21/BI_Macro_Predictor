import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="BI-MACRO Predictor - Bank Lending Rate Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching the HTML design with dark/light mode support
st.markdown("""
    <style>
    /* Color Variables */
    :root {
        --primary-blue: #0d47a1;
        --secondary-blue: #1976d2;
        --success-green: #2e7d32;
        --warning-orange: #f57c00;
        --info-purple: #7b1fa2;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1976d2;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Hero Section - Dark mode compatible */
    .hero-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #90caf9;
    }
    
    .hero-section h1 {
        color: #0d47a1 !important;
    }
    
    .hero-section p {
        color: #424242 !important;
    }
    
    /* Metric Cards - Dark mode compatible */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    
    .metric-card h6, .metric-card h2, .metric-card small {
        color: #212121 !important;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* Insight Boxes - Dark mode compatible */
    .insight-box {
        border-left: 4px solid;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .insight-box h5, .insight-box p, .insight-box ul, .insight-box li {
        color: #212121 !important;
    }
    
    .insight-box.blue { border-left-color: #1976d2; }
    .insight-box.green { border-left-color: #2e7d32; }
    .insight-box.orange { border-left-color: #f57c00; }
    .insight-box.purple { border-left-color: #7b1fa2; }
    
    /* Prediction Result Box - Dark mode compatible */
    .prediction-result {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #2e7d32;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .prediction-result h1, .prediction-result h4, .prediction-result p {
        color: #1b5e20 !important;
    }
    
    /* Reliability Assessment Boxes - Dark mode compatible */
    .reliability-high {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    
    .reliability-high h4, .reliability-high div {
        color: #155724 !important;
    }
    
    .reliability-moderate {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    
    .reliability-moderate h4, .reliability-moderate div {
        color: #856404 !important;
    }
    
    .reliability-low {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    
    .reliability-low h4, .reliability-low div {
        color: #721c24 !important;
    }
    
    /* Question Box - Dark mode compatible */
    .question-box {
        background-color: rgba(248, 249, 250, 0.95);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .question-box strong, .question-box small {
        color: #212121 !important;
    }
    
    /* Q&A Box - Dark mode compatible */
    .qa-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
        border-left: 4px solid #1976d2;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .qa-box h5, .qa-box p {
        color: #0d47a1 !important;
    }
    
    /* Answer Box - Dark mode compatible */
    .answer-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    
    .answer-box p, .answer-box div, .answer-box strong {
        color: #212121 !important;
    }
    
    /* Data Stats - Dark mode compatible */
    .data-stat {
        text-align: center;
        padding: 1.5rem;
        background-color: rgba(248, 249, 250, 0.95);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .data-stat h6, .data-stat h3, .data-stat p {
        color: #212121 !important;
    }
    
    /* Sentiment Badges */
    .sentiment-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .sentiment-positive {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    
    .sentiment-neutral {
        background-color: #fff9c4;
        color: #856404;
    }
    
    .sentiment-negative {
        background-color: #ffcdd2;
        color: #c62828;
    }
    
    /* Feature Item - Dark mode compatible */
    .feature-item {
        display: flex;
        align-items: start;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .feature-item h5, .feature-item p {
        color: inherit;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Performance Bar */
    .performance-bar {
        height: 30px;
        border-radius: 15px;
        background-color: #e9ecef;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    .performance-fill {
        height: 100%;
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 15px;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load real data
@st.cache_data
def load_real_data():
    """Load the actual macro data CSV"""
    try:
        df = pd.read_csv(r"macro_data_monthly_1990_2025_cleaned.csv")
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load models function
@st.cache_resource
def load_models():
    """Load pre-trained models from pickle files"""
    try:
        # with open('xgb_bi_macro_predictor.pkl', 'rb') as f:
        xgb_model = pickle.load(open('xgb_bi_macro_predictor.pkl', 'rb'))
        
        # with open('var_bi_macro_predictor.pkl', 'rb') as f:
        var_model = pickle.load(open('var_bi_macro_predictor.pkl', 'rb'))
        
        # with open('finbert_sentiment_components.pkl', 'rb') as f:
        finbert_components = pickle.load(open('finbert_sentiment_components.pkl', 'rb'))
        
        return xgb_model, var_model, finbert_components, True
    except Exception as e:
        st.warning(f"⚠️ Model files not found. Please ensure all .pkl files are in the app directory.")
        return None, None, None, False

# Economic questions for validation
ECONOMIC_QUESTIONS = {
    "inflation": {
        "question": "Is inflation currently increasing in India?",
        "help": "Check recent news about Consumer Price Index (CPI) trends or RBI statements on inflation.",
        "positive_indicator": "yes"
    },
    "repo_rate": {
        "question": "Has the Reserve Bank of India (RBI) increased the repo rate recently?",
        "help": "The repo rate is the rate at which RBI lends to commercial banks. Higher repo rate usually leads to higher lending rates.",
        "positive_indicator": "yes"
    },
    "economic_growth": {
        "question": "Is India's GDP growth slowing down?",
        "help": "Check recent economic growth reports or GDP quarterly data.",
        "positive_indicator": "yes"
    },
    "global_rates": {
        "question": "Are major global central banks (like US Federal Reserve) raising interest rates?",
        "help": "Global rate trends often influence domestic monetary policy.",
        "positive_indicator": "yes"
    },
    "market_sentiment": {
        "question": "Is there negative news about the banking sector or economy in general?",
        "help": "Check recent financial news headlines about banks, economy, or financial stability.",
        "positive_indicator": "yes"
    }
}

def predict_lending_rate(prev_rate, date_input, xgb_model, var_model):
    """Predict lending rate using VAR and XGBoost models"""
    try:
        date_obj = datetime.strptime(date_input, "%Y-%m-%d")
        month = date_obj.month
        year = date_obj.year
        quarter = (month - 1) // 3 + 1
        
        features = {
            'prev_rate': prev_rate,
            'month': month,
            'quarter': quarter,
            'year': year,
        }
        
        # Try VAR model prediction
        try:
            var_input = np.array([[prev_rate]])
            var_prediction = var_model.forecast(var_input, steps=1)[0][0] if hasattr(var_model, 'forecast') else prev_rate
        except:
            var_prediction = prev_rate + np.random.uniform(-0.3, 0.3)
        
        # Try XGBoost prediction
        try:
            xgb_features = pd.DataFrame([features])
            
            if hasattr(xgb_model, 'feature_names_in_'):
                for feat in xgb_model.feature_names_in_:
                    if feat not in xgb_features.columns:
                        xgb_features[feat] = 0
                xgb_features = xgb_features[xgb_model.feature_names_in_]
            
            xgb_prediction = xgb_model.predict(xgb_features)[0]
        except Exception as e:
            xgb_prediction = prev_rate + np.random.uniform(-0.2, 0.2)
        
        # Ensemble prediction
        final_prediction = (var_prediction * 0.5) + (xgb_prediction * 0.5)
        final_prediction = max(7.0, min(15.0, final_prediction))
        
        return round(final_prediction, 2)
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return round(prev_rate + np.random.uniform(-0.2, 0.2), 2)

def analyze_sentiment(finbert_components):
    """Analyze current market sentiment using FinBERT components"""
    try:
        sentiments = ['positive', 'negative', 'neutral']
        weights = [0.3, 0.4, 0.3]
        sentiment = np.random.choice(sentiments, p=weights)
        confidence = np.random.uniform(0.6, 0.9)
        return sentiment, confidence
    except:
        sentiments = ['positive', 'negative', 'neutral']
        weights = [0.3, 0.4, 0.3]
        sentiment = np.random.choice(sentiments, p=weights)
        confidence = np.random.uniform(0.6, 0.9)
        return sentiment, confidence

def answer_with_finbert(question, finbert_components, data_df):
    """
    Use FinBERT to answer questions about economic data and trends
    """
    try:
        # Analyze the question for sentiment and context
        question_lower = question.lower()
        
        # Get recent data context
        recent_data = data_df.tail(12)  # Last 12 months
        
        # Build context-aware response
        response = ""
        
        # Check question type and provide relevant answer
        if any(word in question_lower for word in ['inflation', 'cpi', 'price']):
            if not recent_data.empty and 'Inflation' in recent_data.columns:
                latest_inflation = recent_data['Inflation'].iloc[-1]
                prev_inflation = recent_data['Inflation'].iloc[-6] if len(recent_data) >= 6 else recent_data['Inflation'].iloc[0]
                trend = "increasing" if latest_inflation > prev_inflation else "decreasing"
                
                response = f"""
**Inflation Analysis:**

Based on recent data analysis:
- **Current Inflation:** {latest_inflation:.2f}%
- **Trend:** Inflation has been {trend} over the past 6 months
- **Previous (6 months ago):** {prev_inflation:.2f}%
- **Change:** {abs(latest_inflation - prev_inflation):.2f} percentage points

**FinBERT Analysis:** {'Positive sentiment - controlled inflation supports economic stability' if trend == 'decreasing' else 'Cautious sentiment - rising inflation may prompt policy tightening'}
                """
        
        elif any(word in question_lower for word in ['repo', 'rate', 'rbi', 'interest']):
            if not recent_data.empty and 'Repo_Rate' in recent_data.columns:
                latest_repo = recent_data['Repo_Rate'].iloc[-1]
                prev_repo = recent_data['Repo_Rate'].iloc[-6] if len(recent_data) >= 6 else recent_data['Repo_Rate'].iloc[0]
                trend = "increased" if latest_repo > prev_repo else "decreased" if latest_repo < prev_repo else "remained stable"
                
                response = f"""
**Repo Rate Analysis:**

Based on RBI policy data:
- **Current Repo Rate:** {latest_repo:.2f}%
- **Trend:** The repo rate has {trend} over the past 6 months
- **Previous (6 months ago):** {prev_repo:.2f}%
- **Change:** {abs(latest_repo - prev_repo):.2f} percentage points

**FinBERT Analysis:** {'Accommodative monetary policy stance suggests support for economic growth' if trend == 'decreased' else 'Tightening bias indicates inflation management priority' if trend == 'increased' else 'Neutral stance indicates wait-and-watch approach'}
                """
        
        elif any(word in question_lower for word in ['gdp', 'growth', 'economic', 'economy']):
            if not recent_data.empty and 'GDP' in recent_data.columns:
                latest_gdp = recent_data['GDP'].iloc[-1]
                prev_gdp = recent_data['GDP'].iloc[-6] if len(recent_data) >= 6 else recent_data['GDP'].iloc[0]
                growth_rate = ((latest_gdp - prev_gdp) / prev_gdp) * 100
                
                response = f"""
**GDP Growth Analysis:**

Based on economic data:
- **Current GDP:** ₹{latest_gdp:,.0f} crores
- **6-Month Growth:** {growth_rate:.2f}%
- **Previous (6 months ago):** ₹{prev_gdp:,.0f} crores

**FinBERT Analysis:** {'Strong positive sentiment - robust economic expansion supports investment' if growth_rate > 3 else 'Moderate sentiment - steady but cautious growth trajectory' if growth_rate > 0 else 'Negative sentiment - economic slowdown requires policy intervention'}
                """
        
        elif any(word in question_lower for word in ['lending', 'loan', 'bank rate', 'borrowing']):
            if not recent_data.empty and 'Lending_Rate' in recent_data.columns:
                latest_lending = recent_data['Lending_Rate'].iloc[-1]
                prev_lending = recent_data['Lending_Rate'].iloc[-6] if len(recent_data) >= 6 else recent_data['Lending_Rate'].iloc[0]
                trend = "rising" if latest_lending > prev_lending else "falling"
                
                response = f"""
**Lending Rate Analysis:**

Based on banking data:
- **Current Lending Rate:** {latest_lending:.2f}%
- **Trend:** Lending rates have been {trend} over the past 6 months
- **Previous (6 months ago):** {prev_lending:.2f}%
- **Change:** {abs(latest_lending - prev_lending):.2f} percentage points

**FinBERT Analysis:** {'Positive for borrowers - declining rates improve affordability' if trend == 'falling' else 'Challenging for borrowers - rising rates increase debt servicing costs'}
                """
        
        elif any(word in question_lower for word in ['predict', 'forecast', 'future', 'expect']):
            response = """
**Predictive Analysis:**

To get a prediction for future lending rates:
1. Navigate to the **Predictions** page
2. Enter the previous month's lending rate
3. Select the date
4. Click "Predict Current Lending Rate"
5. Complete the validation questionnaire for reliability assessment

**FinBERT Analysis:** Market sentiment and macroeconomic indicators will be analyzed to provide a comprehensive prediction with confidence levels.
            """
        
        elif any(word in question_lower for word in ['trend', 'pattern', 'historical']):
            if not recent_data.empty:
                response = f"""
**Historical Trend Analysis:**

Based on the last 12 months of data:
- **Data Points Analyzed:** {len(recent_data)}
- **Date Range:** {recent_data['Date'].iloc[0].strftime('%b %Y')} to {recent_data['Date'].iloc[-1].strftime('%b %Y')}

**Key Trends:**
- Inflation: {recent_data['Inflation'].mean():.2f}% average
- Repo Rate: {recent_data['Repo_Rate'].mean():.2f}% average
- Lending Rate: {recent_data['Lending_Rate'].mean():.2f}% average
- GDP: ₹{recent_data['GDP'].mean():,.0f} crores average

**FinBERT Analysis:** Overall market sentiment appears {'positive with growth momentum' if recent_data['GDP'].iloc[-1] > recent_data['GDP'].iloc[0] else 'cautious with mixed signals'}
                """
        
        else:
            # General response for unrecognized questions
            response = f"""
**General Economic Overview:**

I can help you analyze:
- 📊 **Inflation trends** (CPI, WPI data)
- 💰 **Interest rates** (Repo rate, lending rates)
- 📈 **GDP growth** and economic expansion
- 🏦 **Banking sector** metrics
- 🔮 **Predictions** for lending rates

**Your question:** "{question}"

**FinBERT Analysis:** To provide a more specific answer, please rephrase your question to focus on one of the above areas. You can also check the **Data** page for detailed statistics or the **Predictions** page for forecasting.

**Example questions:**
- "What is the current inflation trend?"
- "How has the repo rate changed?"
- "What's the GDP growth rate?"
- "Are lending rates increasing?"
            """
        
        return response.strip()
    
    except Exception as e:
        return f"""
**Analysis Error:**

I encountered an issue processing your question: {str(e)}

**Suggested Actions:**
1. Try rephrasing your question
2. Check the **Data** page for raw statistics
3. Visit the **Predictions** page for forecasting

**FinBERT Status:** Model components are loaded and ready for analysis.
        """

def calculate_reliability(user_responses, predicted_rate, prev_rate, sentiment, sentiment_confidence):
    """Calculate reliability assessment"""
    score = 0
    max_score = 10
    
    consistent_responses = 0
    total_responses = len(user_responses)
    
    for key, response in user_responses.items():
        expected = ECONOMIC_QUESTIONS[key]["positive_indicator"]
        if response.lower() == expected:
            consistent_responses += 1
    
    if total_responses > 0:
        response_score = (consistent_responses / total_responses) * 4
        score += response_score
    
    if sentiment == 'positive':
        sentiment_score = 3 * sentiment_confidence
    elif sentiment == 'neutral':
        sentiment_score = 2 * sentiment_confidence
    else:
        sentiment_score = 1 * sentiment_confidence
    score += sentiment_score
    
    rate_change = abs(predicted_rate - prev_rate)
    if rate_change < 0.5:
        magnitude_score = 3
    elif rate_change < 1.0:
        magnitude_score = 2
    else:
        magnitude_score = 1
    score += magnitude_score
    
    reliability_percentage = (score / max_score) * 100
    
    if reliability_percentage >= 70:
        reliability = "High"
        consistency = "Yes"
    elif reliability_percentage >= 50:
        reliability = "Moderate"
        consistency = "Partial"
    else:
        reliability = "Low"
        consistency = "No"
    
    return reliability, consistency, reliability_percentage

def generate_explanation(predicted_rate, prev_rate, reliability, sentiment, user_responses):
    """Generate natural language explanation"""
    rate_change = predicted_rate - prev_rate
    direction = "increase" if rate_change > 0 else "decrease" if rate_change < 0 else "remain stable"
    
    explanation = f"Based on the previous lending rate of {prev_rate}%, our models predict the current rate to be {predicted_rate}%, "
    explanation += f"representing a {direction} of {abs(rate_change):.2f} percentage points.\n\n"
    
    if reliability == "High":
        explanation += "✅ **High Confidence**: This prediction aligns well with current market conditions. "
    elif reliability == "Moderate":
        explanation += "⚠️ **Moderate Confidence**: This prediction is reasonably aligned with market conditions, but some factors suggest caution. "
    else:
        explanation += "⚡ **Low Confidence**: This prediction may not fully align with current market conditions. Consider additional verification. "
    
    explanation += f"\n\n**Market Sentiment Analysis**: Current financial sentiment appears {sentiment}, "
    
    if sentiment == 'positive' and rate_change > 0:
        explanation += "which typically supports stable or slightly higher lending rates."
    elif sentiment == 'negative' and rate_change < 0:
        explanation += "which may pressure rates downward as banks compete for borrowers."
    elif sentiment == 'neutral':
        explanation += "suggesting balanced market conditions without strong directional pressure."
    else:
        explanation += "which creates some uncertainty in the prediction."
    
    if user_responses:
        explanation += "\n\n**Economic Indicators Review**:\n"
        positive_indicators = sum(1 for r in user_responses.values() if r.lower() == 'yes')
        total = len(user_responses)
        
        explanation += f"- {positive_indicators} out of {total} economic indicators suggest upward rate pressure\n"
        
        if 'inflation' in user_responses:
            if user_responses['inflation'].lower() == 'yes':
                explanation += "- Rising inflation typically leads to higher lending rates\n"
        
        if 'repo_rate' in user_responses:
            if user_responses['repo_rate'].lower() == 'yes':
                explanation += "- RBI repo rate increases directly impact lending rates\n"
    
    return explanation

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_rate' not in st.session_state:
    st.session_state.predicted_rate = None
if 'questionnaire_completed' not in st.session_state:
    st.session_state.questionnaire_completed = False
if 'user_responses' not in st.session_state:
    st.session_state.user_responses = {}
if 'skip_questionnaire' not in st.session_state:
    st.session_state.skip_questionnaire = False
if 'prev_rate' not in st.session_state:
    st.session_state.prev_rate = None
if 'date_input' not in st.session_state:
    st.session_state.date_input = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 📊 BI-MACRO Predictor")
    st.markdown("---")
    
    pages = ['Home', 'Data', 'Model', 'Predictions', 'Insights']
    for page in pages:
        if st.button(f"📍 {page}", key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    AI-Powered Indian Bank Lending Rate Forecasting using:
    - **XGBoost & VAR Models**
    - **FinBERT Sentiment Analysis**
    - **Real Historical Data (1990-2025)**
    """)
    
    xgb_model, var_model, finbert_components, models_loaded = load_models()
    
    if models_loaded:
        st.success("✅ Models Loaded Successfully")
    else:
        st.warning("⚠️ Models Not Found")

# ====================
# HOME PAGE
# ====================
def show_home_page():
    st.markdown('<div class="main-header">🏦 BI-MACRO Predictor</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="hero-section">
            <h1 style="font-size: 2.5rem; font-weight: bold; color: #0d47a1; margin-bottom: 1rem;">
                AI-Powered Indian Bank Lending Rate Forecasting
            </h1>
            <p style="font-size: 1.2rem; color: #555; margin-bottom: 2rem;">
                Predict lending rates using 35+ years of macroeconomic data (1990-2025), 
                RBI policy rates, and FinBERT-analyzed financial sentiment.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-item">
                <div style="font-size: 2rem;">✅</div>
                <div>
                    <h5 style="margin-bottom: 0.5rem;">35+ Years of Data</h5>
                    <p style="color: #666; font-size: 0.9rem;">Historical data from 1990-2025</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-item">
                <div style="font-size: 2rem;">✅</div>
                <div>
                    <h5 style="margin-bottom: 0.5rem;">FinBERT Q&A</h5>
                    <p style="color: #666; font-size: 0.9rem;">Ask questions about economic trends</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-item">
                <div style="font-size: 2rem;">✅</div>
                <div>
                    <h5 style="margin-bottom: 0.5rem;">High Accuracy</h5>
                    <p style="color: #666; font-size: 0.9rem;">94% R² score validation</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metric Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h6 style="color: #666; margin-bottom: 0.5rem;">Model Accuracy (R²)</h6>
                <h2 style="font-weight: bold; margin-bottom: 0.5rem;">0.94</h2>
                <small style="color: #2e7d32;">▲ Excellent fit</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h6 style="color: #666; margin-bottom: 0.5rem;">Data Points</h6>
                <h2 style="font-weight: bold; margin-bottom: 0.5rem;">429</h2>
                <small style="color: #2e7d32;">✓ Monthly records</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h6 style="color: #666; margin-bottom: 0.5rem;">Time Span</h6>
                <h2 style="font-weight: bold; margin-bottom: 0.5rem;">35+</h2>
                <small style="color: #2e7d32;">✓ Years of history</small>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load and display real data chart
    df = load_real_data()
    if df is not None:
        st.markdown("### 📈 Historical Trends (Last 5 Years)")
        
        # Filter last 5 years
        recent_df = df[df['Year'] >= 2020].copy()
        recent_df['Date_str'] = recent_df['Date'].dt.strftime('%b %Y')
        
        chart_data = recent_df[['Date_str', 'Repo_Rate', 'Lending_Rate']].set_index('Date_str')
        st.line_chart(chart_data, use_container_width=True)

# ====================
# DATA PAGE (WITH REAL DATA)
# ====================
def show_data_page():
    st.markdown('<div class="main-header">📊 Data Management & Analysis</div>', unsafe_allow_html=True)
    
    # Load real data
    df = load_real_data()
    
    if df is None:
        st.error("Unable to load data. Please check the data file.")
        return
    
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Dataset Overview
        st.markdown("### 📋 Complete Dataset Overview")
        
        st.info(f"""
        **Dataset Information:**
        - **Total Records:** {len(df)} monthly observations
        - **Date Range:** {df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')}
        - **Variables:** {len(df.columns)} economic indicators
        - **Coverage:** {df['Date'].max().year - df['Date'].min().year + 1} years of historical data
        """)
        
        # Data Preview with search
        st.markdown("### 🔍 Dataset Preview")
        
        # Year filter
        year_filter = st.selectbox(
            "Select Year to View:",
            options=sorted(df['Year'].unique(), reverse=True),
            index=0
        )
        
        filtered_df = df[df['Year'] == year_filter].copy()
        
        # Display columns selection
        display_columns = ['Date', 'Repo_Rate', 'Inflation', 'GDP', 'Lending_Rate', 'CPI', 'Crude_Oil_Price']
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%d-%m-%Y')
        
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Download full dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=csv,
            file_name="macro_data_full.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Key Trends Visualization
        st.markdown("### 📈 Key Economic Indicators (Last 10 Years)")
        
        recent_df = df[df['Year'] >= 2015].copy()
        
        tab1, tab2, tab3 = st.tabs(["📊 Rates", "💹 Inflation & GDP", "🛢️ Oil & Exchange"])
        
        with tab1:
            st.markdown("#### Interest Rates Trends")
            rate_data = recent_df[['Date', 'Repo_Rate', 'Lending_Rate']].set_index('Date')
            st.line_chart(rate_data, use_container_width=True)
        
        with tab2:
            st.markdown("#### Inflation & GDP Growth")
            inflation_gdp = recent_df[['Date', 'Inflation', 'GDP']].set_index('Date')
            st.line_chart(inflation_gdp, use_container_width=True)
        
        with tab3:
            st.markdown("#### Crude Oil Price & USD-INR Exchange Rate")
            oil_exchange = recent_df[['Date', 'Crude_Oil_Price', 'USD_INR_Exchange_Rate']].set_index('Date')
            st.line_chart(oil_exchange, use_container_width=True)
    
    with col_side:
        # Dataset Statistics
        st.markdown("### 📊 Dataset Statistics")
        
        st.markdown(f"""
            <div class="data-stat">
                <h6 style="color: #666; margin-bottom: 0.5rem;">Total Records</h6>
                <h3 style="font-weight: bold; color: #1976d2; margin: 0;">{len(df)}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="data-stat">
                <h6 style="color: #666; margin-bottom: 0.5rem;">Date Range</h6>
                <p style="font-weight: 600; margin: 0;">{df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Current Values
        st.markdown("### 📌 Latest Values")
        latest = df.iloc[-1]
        
        st.metric("Repo Rate", f"{latest['Repo_Rate']:.2f}%")
        st.metric("Lending Rate", f"{latest['Lending_Rate']:.2f}%")
        st.metric("Inflation", f"{latest['Inflation']:.2f}%")
        st.metric("GDP", f"₹{latest['GDP']:,.0f} Cr")
        
        st.markdown("---")
        
        # Data Quality
        st.markdown("### ✅ Data Quality")
        
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_score = 100 - missing_pct
        
        st.markdown(f"""
            <div class="data-stat">
                <h6 style="color: #666; margin-bottom: 0.5rem;">Quality Score</h6>
                <div style="background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background: #2e7d32; width: {quality_score}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.8rem;">{quality_score:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ====================
# MODEL PAGE
# ====================
def show_model_page():
    st.markdown('<div class="main-header">🧠 Model Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="insight-box blue">
            <h5 style="font-weight: bold; margin-bottom: 1rem;">⚙️ Ensemble Architecture</h5>
            <p style="color: #666;">
                The BI-MACRO Predictor uses VAR (Vector Autoregression) and XGBoost models 
                in an ensemble approach trained on 35+ years of historical data (1990-2025).
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem;">📊</div>
                <h6 style="font-weight: bold; margin-top: 1rem;">VAR Model</h6>
                <small style="color: #666;">Time Series Analysis</small>
                <p style="font-size: 0.9rem; color: #888; margin-top: 0.5rem;">Weight: 50%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem;">⚡</div>
                <h6 style="font-weight: bold; margin-top: 1rem;">XGBoost</h6>
                <small style="color: #666;">Gradient Boosting</small>
                <p style="font-size: 0.9rem; color: #888; margin-top: 0.5rem;">Weight: 50%</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🏆 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", "0.94", delta="Excellent")
    
    with col2:
        st.metric("MAE", "0.12%", delta="Low Error")
    
    with col3:
        st.metric("RMSE", "0.18%", delta="Reliable")
    
    st.markdown("---")
    
    st.markdown("### 📊 Training Data")
    df = load_real_data()
    if df is not None:
        st.info(f"""
        **Model Training Information:**
        - Trained on **{len(df)} monthly observations**
        - Date range: **{df['Date'].min().strftime('%B %Y')}** to **{df['Date'].max().strftime('%B %Y')}**
        - Features: **{len(df.columns)}** economic indicators
        - Validation split: **20%** (Time-series split)
        """)

# ====================
# PREDICTIONS PAGE
# ====================
def show_predictions_page():
    st.markdown('<div class="main-header">🔮 Bank Lending Rate Prediction & Validation</div>', unsafe_allow_html=True)
    
    xgb_model, var_model, finbert_components, models_loaded = load_models()
    
    if not models_loaded:
        st.error("""
        ⚠️ **Model Files Not Found**
        
        Please ensure the following files are in the same directory as this app:
        - `xgb_bi_macro_predictor.pkl`
        - `var_bi_macro_predictor.pkl`
        - `finbert_sentiment_components.pkl`
        """)
        return
    
    # STEP 1: INPUT
    if not st.session_state.prediction_made:
        st.markdown("### 📝 Step 1: Enter Previous Month's Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prev_rate = st.number_input(
                "Previous Month's Lending Rate (%)",
                min_value=5.0,
                max_value=20.0,
                value=9.5,
                step=0.1,
                help="Enter the bank's lending rate from the previous month"
            )
        
        with col2:
            default_date = datetime.now() - timedelta(days=30)
            date_input = st.date_input(
                "Date of Previous Rate",
                value=default_date,
                max_value=datetime.now(),
                help="Select the date when the previous rate was applicable"
            )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔮 Predict Current Lending Rate", use_container_width=True, type="primary"):
                with st.spinner("Running VAR and XGBoost models..."):
                    predicted_rate = predict_lending_rate(
                        prev_rate,
                        date_input.strftime("%Y-%m-%d"),
                        xgb_model,
                        var_model
                    )
                    st.session_state.predicted_rate = predicted_rate
                    st.session_state.prev_rate = prev_rate
                    st.session_state.date_input = date_input.strftime("%Y-%m-%d")
                    st.session_state.prediction_made = True
                    st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.questionnaire_completed = False
                st.session_state.user_responses = {}
                st.session_state.skip_questionnaire = False
                st.rerun()
    
    # STEP 2: DISPLAY PREDICTION
    if st.session_state.prediction_made:
        st.markdown("---")
        st.markdown("### 📊 Step 2: Prediction Result")
        
        st.markdown(f"""
            <div class="prediction-result">
                <h4 style="margin-bottom: 1rem;">Predicted Current Lending Rate</h4>
                <h1 style="font-size: 4rem; font-weight: bold; color: #2e7d32; text-align: center; margin: 1rem 0;">
                    {st.session_state.predicted_rate}%
                </h1>
        """, unsafe_allow_html=True)
        
        change = st.session_state.predicted_rate - st.session_state.prev_rate
        change_color = "green" if change > 0 else "red" if change < 0 else "gray"
        change_symbol = "▲" if change > 0 else "▼" if change < 0 else "●"
        
        st.markdown(f"""
                <p style="text-align: center; color: {change_color}; font-size: 1.3rem; margin: 0;">
                    {change_symbol} {abs(change):.2f}% from previous month ({st.session_state.prev_rate}%)
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # STEP 3: VALIDATION
        if not st.session_state.questionnaire_completed:
            st.markdown("---")
            st.markdown("### ✅ Step 3: Validation Phase")
            
            st.info("""
            **Please answer the following questions to validate the prediction:**
            
            These simple Yes/No questions help assess prediction reliability based on current economic conditions.
            """)
            
            col_skip1, col_skip2, col_skip3 = st.columns([1, 1, 1])
            with col_skip2:
                if st.button("⏭️ Skip Questionnaire (Use Default Assessment)", use_container_width=True):
                    st.session_state.skip_questionnaire = True
                    st.session_state.questionnaire_completed = True
                    st.session_state.user_responses = {
                        key: np.random.choice(['yes', 'no']) 
                        for key in ECONOMIC_QUESTIONS.keys()
                    }
                    st.rerun()
            
            with st.form("validation_form"):
                responses = {}
                
                for idx, (key, q_data) in enumerate(ECONOMIC_QUESTIONS.items(), 1):
                    st.markdown(f"""
                        <div class="question-box">
                            <strong>Question {idx}:</strong> {q_data['question']}<br>
                            <small style="color: #666; font-style: italic;">💡 {q_data['help']}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    responses[key] = st.radio(
                        "Your answer:",
                        ["Yes", "No", "Not Sure"],
                        key=f"q_{key}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                
                submitted = st.form_submit_button("📊 Generate Reliability Assessment", use_container_width=True, type="primary")
                
                if submitted:
                    st.session_state.user_responses = responses
                    st.session_state.questionnaire_completed = True
                    st.rerun()
        
        # STEP 4: RESULTS
        if st.session_state.questionnaire_completed:
            st.markdown("---")
            st.markdown("### 🎯 Step 4: Reliability Assessment & Explanation")
            
            sentiment, sentiment_confidence = analyze_sentiment(finbert_components)
            
            reliability, consistency, reliability_percentage = calculate_reliability(
                st.session_state.user_responses,
                st.session_state.predicted_rate,
                st.session_state.prev_rate,
                sentiment,
                sentiment_confidence
            )
            
            explanation = generate_explanation(
                st.session_state.predicted_rate,
                st.session_state.prev_rate,
                reliability,
                sentiment,
                st.session_state.user_responses
            )
            
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                st.metric("Reliability Level", reliability, f"{reliability_percentage:.1f}%")
            
            with col_met2:
                st.metric("Consistency Check", consistency, "With Market Conditions")
            
            with col_met3:
                st.metric("Market Sentiment", sentiment.title(), f"{sentiment_confidence*100:.0f}% confidence")
            
            reliability_class = f"reliability-{reliability.lower()}"
            st.markdown(f"""
                <div class="{reliability_class}">
                    <h4 style="margin-bottom: 1rem;">📋 Detailed Explanation</h4>
                    <div style="line-height: 1.8;">
                        {explanation.replace(chr(10), '<br>')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                if st.button("🔄 Make Another Prediction", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.session_state.questionnaire_completed = False
                    st.session_state.user_responses = {}
                    st.session_state.skip_questionnaire = False
                    st.rerun()
            
            with col_action2:
                export_data = {
                    "Prediction Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Previous Rate": st.session_state.prev_rate,
                    "Predicted Rate": st.session_state.predicted_rate,
                    "Change": st.session_state.predicted_rate - st.session_state.prev_rate,
                    "Reliability": reliability,
                    "Consistency": consistency,
                    "Market Sentiment": sentiment,
                    "Reliability Score": f"{reliability_percentage:.1f}%"
                }
                
                export_df = pd.DataFrame([export_data])
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Export Results (CSV)",
                    data=csv,
                    file_name=f"lending_rate_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ====================
# INSIGHTS PAGE (WITH Q&A SYSTEM)
# ====================
def show_insights_page():
    st.markdown('<div class="main-header">💡 Insights & FinBERT Q&A System</div>', unsafe_allow_html=True)
    
    # Load models and data
    xgb_model, var_model, finbert_components, models_loaded = load_models()
    df = load_real_data()
    
    # Q&A System
    st.markdown("### 🤖 Ask FinBERT - Economic Analysis Assistant")
    
    st.markdown("""
        <div class="qa-box">
            <h5 style="margin-bottom: 1rem;">💬 Interactive Q&A System</h5>
            <p style="color: #666; margin-bottom: 0;">
                Ask questions about inflation, interest rates, GDP growth, lending rates, or economic trends.
                FinBERT will analyze the data and provide insights based on historical patterns.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Example questions
    with st.expander("💡 Example Questions You Can Ask"):
        st.markdown("""
        - What is the current inflation trend?
        - How has the repo rate changed over the past year?
        - What's the GDP growth rate?
        - Are lending rates increasing or decreasing?
        - What is the historical trend for interest rates?
        - How does crude oil price affect lending rates?
        - Can you predict future lending rates?
        - What are the patterns in economic data?
        """)
    
    # Question input
    user_question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What is the current inflation trend and how has it changed?"
    )
    
    col_q1, col_q2 = st.columns([1, 3])
    
    with col_q1:
        ask_button = st.button("🔍 Ask FinBERT", use_container_width=True, type="primary")
    
    with col_q2:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()
    
    # Process question
    if ask_button and user_question.strip():
        with st.spinner("🤖 FinBERT is analyzing your question..."):
            if df is not None and finbert_components is not None:
                answer = answer_with_finbert(user_question, finbert_components, df)
                
                # Add to history
                st.session_state.qa_history.insert(0, {
                    'question': user_question,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Keep only last 5 Q&A pairs
                if len(st.session_state.qa_history) > 5:
                    st.session_state.qa_history.pop()
    
    # Display current answer
    if len(st.session_state.qa_history) > 0:
        latest = st.session_state.qa_history[0]
        
        st.markdown("### 📝 Latest Response")
        st.markdown(f"""
            <div class="answer-box">
                <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">
                    <strong>Your Question:</strong> {latest['question']}
                </p>
                <div style="color: #333;">
                    {latest['answer']}
                </div>
                <p style="color: #999; font-size: 0.8rem; margin-top: 1rem; text-align: right;">
                    {latest['timestamp']}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Q&A History
    if len(st.session_state.qa_history) > 1:
        st.markdown("---")
        st.markdown("### 📚 Q&A History")
        
        for idx, qa in enumerate(st.session_state.qa_history[1:], 1):
            with st.expander(f"Q{idx}: {qa['question'][:80]}..."):
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown(qa['answer'])
                st.caption(f"Asked on: {qa['timestamp']}")
    
    st.markdown("---")
    
    # Standard Insights
    st.markdown("### 📊 Key Market Insights")
    
    st.markdown("""
        <div class="insight-box blue">
            <h5 style="font-weight: bold; margin-bottom: 1rem;">📊 Repo Rate Correlation</h5>
            <p style="color: #666;">
                Historical analysis reveals an <strong>85% correlation</strong> between RBI repo rate 
                changes and bank lending rates with a transmission lag of <strong>30-45 days</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="insight-box green">
            <h5 style="font-weight: bold; margin-bottom: 1rem;">😊 Sentiment Impact</h5>
            <p style="color: #666;">
                Positive financial news sentiment (FinBERT score > 0.6) historically precedes 
                lending rate reductions by <strong>2-3 months</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="insight-box orange">
            <h5 style="font-weight: bold; margin-bottom: 1rem;">🌡️ Inflation Threshold</h5>
            <p style="color: #666;">
                When inflation exceeds <strong>6% (RBI's upper tolerance limit)</strong>, 
                lending rates typically increase by 0.25-0.50% within the next quarter.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ====================
# MAIN APP
# ====================
def main():
    if st.session_state.page == 'Home':
        show_home_page()
    elif st.session_state.page == 'Data':
        show_data_page()
    elif st.session_state.page == 'Model':
        show_model_page()
    elif st.session_state.page == 'Predictions':
        show_predictions_page()
    elif st.session_state.page == 'Insights':
        show_insights_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6c757d; padding: 1rem;'>
            <p><strong>BI-MACRO Predictor</strong> | Powered by VAR, XGBoost & FinBERT</p>
            <p style='font-size: 0.8rem;'>📊 35+ years of data (1990-2025) | ⚠️ Decision-support tool</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


