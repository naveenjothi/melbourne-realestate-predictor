# 🏠 Melbourne Housing Price Predictor

An interactive Streamlit web application that predicts property prices in Melbourne using machine learning.

## ✨ Features

- **Interactive UI**: Modern, responsive design with real-time predictions
- **Comprehensive Instructions**: Built-in guide on how to use the app
- **Visual Analytics**: Charts and metrics to understand price factors
- **What-If Analysis**: Interactive scenarios to see how changes affect prices
- **Market Context**: Compare predictions with suburb averages and trends

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository** (if not already done)
2. **Set up virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📱 How to Use the App

### Step 1: Property Details Input

- Use the **sidebar** to input your property details:
  - **Location**: Select from Richmond, South Yarra, or St Kilda
  - **Property Type**: Choose between House, Unit, or Townhouse
  - **Specifications**: Set rooms, bathrooms, and car spaces
  - **Size**: Input land size and building area
  - **Distance**: Set distance from Melbourne CBD

### Step 2: View Your Prediction

- The **main area** displays:
  - **Property Summary**: Overview of your inputs
  - **Price Prediction**: Estimated property value
  - **Price Insights**: Price per square meter and estimated range
  - **Market Context**: How your property compares to suburb averages

### Step 3: Interactive Analysis

Explore the three tabs at the bottom:

1. **📊 Price Factors**: Visual breakdown of what affects property prices
2. **🔄 What-If Analysis**: See how adding rooms or bathrooms impacts price
3. **ℹ️ About**: Learn about the model and its limitations

## 🎯 Key Features Explained

### Real-time Updates

- Predictions update automatically as you change input parameters
- No need to click "Predict" - everything is instant!

### Visual Feedback

- **Metrics Cards**: Quick overview of key property details
- **Price Display**: Large, prominent price with gradient styling
- **Charts**: Interactive visualizations of price factors

### Market Intelligence

- **Suburb Comparisons**: See if your property is above or below area average
- **Price Range**: Get confidence intervals (±10%) for your estimate
- **Trend Information**: Current market trends for each suburb

## 🤖 Model Information

- **Algorithm**: XGBoost Regression
- **Features**: Location, property type, size, rooms, bathrooms, car spaces, distance to CBD
- **Training Data**: Historical Melbourne housing market data
- **Accuracy**: Provides estimates based on historical patterns

## ⚠️ Important Notes

- **Estimates Only**: Use for preliminary assessments, not final valuations
- **Market Conditions**: Actual prices may vary due to current market conditions
- **Professional Advice**: Consult real estate professionals for investment decisions
- **Data Currency**: Model trained on historical data - market conditions change

## 🛠️ Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **XGBoost**: Machine learning model
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Data preprocessing

### Model Assets Required

The app requires these pre-trained model files:

- `xgb_model.pkl`: Trained XGBoost model
- `scaler.pkl`: Feature scaler
- `model_columns.pkl`: Model column names
- `numerical_columns.pkl`: Numerical feature names

## 🎨 UI Enhancements

### Modern Styling

- **Gradient Headers**: Eye-catching titles with color gradients
- **Card Layout**: Clean, organized information display
- **Responsive Design**: Works well on different screen sizes

### Interactive Elements

- **Tooltips**: Helpful hints for each input field
- **Expandable Sections**: Detailed information when needed
- **Tabbed Interface**: Organized analysis tools

### Visual Feedback

- **Color-coded Metrics**: Easy-to-read property information
- **Progress Indicators**: Visual feedback for user actions
- **Status Messages**: Clear success and information messages

## 🚀 Running the App

### Development Mode

```bash
streamlit run app.py
```

### Production Mode

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### With Custom Configuration

```bash
streamlit run app.py --theme.primaryColor "#667eea"
```

## 📊 Sample Usage

1. **Basic Prediction**:

   - Select "Richmond" as suburb
   - Set 3 rooms, 2 bathrooms
   - Choose "House" as property type
   - Set 5km from CBD
   - Input 400m² land, 150m² building

2. **What-If Analysis**:

   - Use the "What-If Analysis" tab
   - Click "See +1 Room Impact" to see price change
   - Compare different scenarios

3. **Market Research**:
   - Try different suburbs to compare prices
   - Adjust property types to see value differences
   - Use distance slider to understand location premium

## 🤝 Support

For issues or questions:

1. Check the "About" tab in the app for model information
2. Review the built-in instructions in the app
3. Ensure all model files are present in the project directory

---

**Built with ❤️ using Streamlit and XGBoost**
