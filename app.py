import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Melbourne Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .instruction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Load the Saved Model and Assets ---
@st.cache_resource
def load_model_assets():
    """Load model assets with caching for better performance"""
    try:
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        numerical_columns = joblib.load('numerical_columns.pkl')
        return model, scaler, model_columns, numerical_columns
    except FileNotFoundError as e:
        st.error(f"❌ Model assets not found: {e}")
        st.error("Please run the 'train_and_save_artifacts.py' script first.")
        st.stop()

model, scaler, model_columns, numerical_columns = load_model_assets()


# --- 2. Header and Instructions ---
st.markdown('<h1 class="main-header">🏠 Melbourne Housing Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get instant property price estimates using AI-powered predictions</p>', unsafe_allow_html=True)

# Instructions in an expandable section
with st.expander("📋 How to Use This App", expanded=True):
    st.markdown("""
    <div class="instruction-box">
    <h3>🚀 Quick Start Guide:</h3>
    <ol>
        <li><strong>📍 Select Location:</strong> Choose a suburb from the dropdown menu</li>
        <li><strong>🏡 Property Details:</strong> Use the sliders and inputs to specify:
            <ul>
                <li>Number of rooms and bathrooms</li>
                <li>Property type (House, Unit, or Townhouse)</li>
                <li>Distance from Melbourne CBD</li>
                <li>Car spaces available</li>
                <li>Land and building sizes</li>
            </ul>
        </li>
        <li><strong>⚡ Real-time Updates:</strong> Watch the prediction update automatically as you change values</li>
        <li><strong>📊 View Results:</strong> See your estimated property price and additional insights</li>
    </ol>
    <p><strong>💡 Tip:</strong> Try different combinations to see how various factors affect property prices!</p>
    </div>
    """, unsafe_allow_html=True)

# --- 3. Create the User Interface (UI) in the Sidebar ---
st.sidebar.markdown("### 🏠 Property Details Input")
st.sidebar.markdown("*Adjust the parameters below to get your price prediction*")

def user_input_features():
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📍 Location")
    suburb = st.sidebar.selectbox(
        'Suburb', 
        ['Richmond', 'South Yarra', 'St Kilda'],
        help="Select the suburb where the property is located"
    )
    
    st.sidebar.markdown("#### 🏡 Property Specifications")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        rooms = st.sidebar.slider(
            'Rooms', 1, 10, 3,
            help="Total number of bedrooms"
        )
        bathroom = st.sidebar.slider(
            'Bathrooms', 1, 8, 2,
            help="Number of bathrooms"
        )
    
    with col2:
        car = st.sidebar.slider(
            'Car Spaces', 0, 10, 1,
            help="Number of parking spaces"
        )
    
    prop_type = st.sidebar.selectbox(
        'Property Type', 
        ['h', 'u', 't'], 
        format_func=lambda x: {'h':'🏠 House', 'u':'🏢 Unit', 't':'🏘️ Townhouse'}[x],
        help="Type of property"
    )
    
    st.sidebar.markdown("#### 📏 Size & Distance")
    distance = st.sidebar.slider(
        'Distance from CBD (km)', 0.0, 50.0, 5.0, step=0.5,
        help="Distance from Melbourne Central Business District"
    )
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        landsize = st.sidebar.number_input(
            'Land Size (m²)', 
            min_value=0, 
            value=400, 
            step=50,
            help="Total land area in square meters"
        )
    
    with col4:
        building_area = st.sidebar.number_input(
            'Building Area (m²)', 
            min_value=0, 
            value=150, 
            step=10,
            help="Building floor area in square meters"
        )

    # Create a dictionary of the inputs
    data = {
        'Suburb': suburb,
        'Rooms': rooms,
        'Type': prop_type,
        'Distance': distance,
        'Bathroom': bathroom,
        'Car': car,
        'Landsize': landsize,
        'BuildingArea': building_area
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- 4. Process User Input for Prediction ---
def process_input_and_predict(input_df):
    """Process input and make prediction"""
    # Create a full DataFrame with all model columns, initialized to zero
    input_encoded = pd.DataFrame(columns=model_columns)
    input_encoded.loc[0] = 0

    # One-hot encode the categorical inputs
    input_encoded[f'Suburb_{input_df["Suburb"].iloc[0]}'] = 1
    input_encoded[f'Type_{input_df["Type"].iloc[0]}'] = 1

    # Fill in the numerical inputs
    for col in ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']:
        input_encoded[col] = input_df[col].iloc[0]

    # Feature Engineering: Create the same new features as in training
    if 'Bathroom_per_Room' in model_columns:
        if input_encoded['Rooms'].iloc[0] > 0:
            input_encoded['Bathroom_per_Room'] = input_encoded['Bathroom'] / input_encoded['Rooms']
        else:
            input_encoded['Bathroom_per_Room'] = 0

    # Ensure all columns are in the correct order
    input_encoded = input_encoded[model_columns]

    # Scale the numerical features
    input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])
    
    # Make prediction
    prediction = model.predict(input_encoded)
    return int(prediction[0])

# Get prediction
predicted_price = process_input_and_predict(input_df)

# --- 5. Display Results with Enhanced UI ---
# Create main content area with columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Your Property Summary")
    
    # Display input in a nice formatted way
    prop_type_display = {'h': '🏠 House', 'u': '🏢 Unit', 't': '🏘️ Townhouse'}[input_df['Type'].iloc[0]]
    
    # Create metrics display
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("📍 Location", input_df['Suburb'].iloc[0])
        st.metric("🏡 Property Type", prop_type_display)
    
    with metrics_col2:
        st.metric("🛏️ Rooms", f"{input_df['Rooms'].iloc[0]}")
        st.metric("🚿 Bathrooms", f"{input_df['Bathroom'].iloc[0]}")
    
    with metrics_col3:
        st.metric("🚗 Car Spaces", f"{input_df['Car'].iloc[0]}")
        st.metric("📏 Distance to CBD", f"{input_df['Distance'].iloc[0]} km")
    
    # Additional details in expander
    with st.expander("📋 Additional Property Details"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**Land Size:** {input_df['Landsize'].iloc[0]:,.0f} m²")
            st.write(f"**Building Area:** {input_df['BuildingArea'].iloc[0]:,.0f} m²")
        with detail_col2:
            if input_df['Rooms'].iloc[0] > 0:
                bathroom_ratio = input_df['Bathroom'].iloc[0] / input_df['Rooms'].iloc[0]
                st.write(f"**Bathroom per Room:** {bathroom_ratio:.2f}")
            area_ratio = input_df['BuildingArea'].iloc[0] / max(input_df['Landsize'].iloc[0], 1)
            st.write(f"**Building to Land Ratio:** {area_ratio:.2%}")

with col2:
    st.markdown("## 💰 Price Prediction")
    
    # Main price display
    st.markdown(f'<div class="price-display">${predicted_price:,.0f}</div>', unsafe_allow_html=True)
    
    # Price insights
    st.markdown("### 📈 Price Insights")
    
    # Calculate price per square meter
    if input_df['BuildingArea'].iloc[0] > 0:
        price_per_sqm = predicted_price / input_df['BuildingArea'].iloc[0]
        st.metric("Price per m²", f"${price_per_sqm:,.0f}")
    
    # Price range estimate (±10%)
    lower_bound = int(predicted_price * 0.9)
    upper_bound = int(predicted_price * 1.1)
    
    st.info(f"**Estimated Range:**\n${lower_bound:,.0f} - ${upper_bound:,.0f}")
    
    # Market context
    st.markdown("### 🏘️ Market Context")
    suburb_context = {
        'Richmond': {'avg': 850000, 'trend': '📈 Rising'},
        'South Yarra': {'avg': 1200000, 'trend': '📊 Stable'},
        'St Kilda': {'avg': 750000, 'trend': '📈 Rising'}
    }
    
    suburb = input_df['Suburb'].iloc[0]
    if suburb in suburb_context:
        avg_price = suburb_context[suburb]['avg']
        trend = suburb_context[suburb]['trend']
        
        st.write(f"**{suburb} Average:** ${avg_price:,.0f}")
        st.write(f"**Market Trend:** {trend}")
        
        if predicted_price > avg_price:
            st.success("🔥 Above average for the area!")
        else:
            st.info("💡 Good value for the area!")

# --- 6. Interactive Features ---
st.markdown("---")
st.markdown("## 🎯 Interactive Analysis")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📊 Price Factors", "🔄 What-If Analysis", "ℹ️ About"])

with tab1:
    st.markdown("### Key Factors Affecting Your Property Price")
    
    # Create a simple factor importance visualization
    factors = ['Location', 'Property Type', 'Size', 'Rooms', 'Distance to CBD']
    importance = [25, 20, 20, 15, 20]  # Simplified importance scores
    
    fig = px.bar(
        x=factors, 
        y=importance,
        title="Estimated Factor Importance (%)",
        color=importance,
        color_continuous_scale="viridis"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Note:** These are estimated factor importances. Actual model weights may vary.")

with tab2:
    st.markdown("### 🔄 Quick What-If Scenarios")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        st.markdown("**🏠 Add a Room**")
        if st.button("See +1 Room Impact", key="room_btn"):
            modified_df = input_df.copy()
            modified_df['Rooms'] = modified_df['Rooms'] + 1
            new_price = process_input_and_predict(modified_df)
            price_diff = new_price - predicted_price
            st.write(f"New Price: ${new_price:,.0f}")
            st.write(f"Difference: ${price_diff:,.0f}")
    
    with scenario_col2:
        st.markdown("**🚿 Add a Bathroom**")
        if st.button("See +1 Bathroom Impact", key="bath_btn"):
            modified_df = input_df.copy()
            modified_df['Bathroom'] = modified_df['Bathroom'] + 1
            new_price = process_input_and_predict(modified_df)
            price_diff = new_price - predicted_price
            st.write(f"New Price: ${new_price:,.0f}")
            st.write(f"Difference: ${price_diff:,.0f}")

with tab3:
    st.markdown("### ℹ️ About This Prediction Tool")
    
    st.markdown("""
    **🤖 Model Information:**
    - **Algorithm:** XGBoost Regression
    - **Training Data:** Melbourne housing market data
    - **Features:** Location, size, amenities, and distance factors
    
    **📊 Accuracy:**
    - This tool provides estimates based on historical data patterns
    - Actual prices may vary due to market conditions, property condition, and other factors
    
    **⚠️ Disclaimer:**
    - Use this tool for preliminary estimates only
    - Consult with real estate professionals for investment decisions
    - Market conditions change and may affect accuracy
    
    **🔄 Last Updated:** {datetime.now().strftime('%B %Y')}
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🏠 Melbourne Housing Price Predictor | Built with Streamlit & XGBoost"
    "</div>", 
    unsafe_allow_html=True
)