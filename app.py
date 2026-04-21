import streamlit as st
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# ==========================================
# 1. API KEY SETUP
# ==========================================
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set the GOOGLE_API_KEY in Streamlit Secrets.")

# ==========================================
# 2. LOAD MODEL & REFINED PREDICTION LOGIC
# ==========================================
@st.cache_resource
def load_prop_model():
    model = xgb.XGBRegressor()
    model_path = 'prop_model_updated.json' 
    
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
        except Exception as e:
            print(f"Model load failed: {e}")
    return model

xgb_model = load_prop_model()

def get_refined_prediction(bhk, sqft, year):
    features = np.array([[bhk, sqft, year]])
    try:
        raw_val = xgb_model.predict(features)[0]
        temp_val = raw_val
        while temp_val > 150: 
            temp_val = temp_val / 10.0
            
        final_price = max(temp_val, (sqft * 3500) / 100000)
        final_price = final_price + (bhk * 8.0)
        return round(final_price, 2)
    except:
        return 65.0 

@tool
def property_price_predictor(bhk: int, sqft: int, year_built: int):
    """Predicts house price. Input: bhk, sqft, year_built."""
    prediction = get_refined_prediction(bhk, sqft, year_built)
    return f"STRICT_VALUATION: {prediction} Lakhs (INR). Do not convert this to Crores."

@tool
def investment_advisor(bhk: int, sqft: int, current_listing_price: float, year_built: int):
    """
    Analyzes property investment. 
    IMPORTANT: current_listing_price MUST be in Lakhs (e.g., 58.0 for 58 Lakhs).
    """
    val = current_listing_price
    if val > 5000: 
        val = val / 100000.0
    
    fair_market_value = get_refined_prediction(bhk, sqft, year_built)
    future_value = get_refined_prediction(bhk, sqft, year_built + 5) * 1.15 
    
    is_underpriced = val < fair_market_value
    profit_potential = future_value - val
    roi = (profit_potential / val) * 100
    
    return (
        f"CRITICAL_DATA: \n"
        f"- Market Value: {fair_market_value} Lakhs\n"
        f"- Your Price: {val} Lakhs\n"
        f"- 5-Year Future Value: {round(future_value, 2)} Lakhs\n"
        f"- ROI: {round(roi, 2)}%\n"
        f"- Underpriced: {'YES' if is_underpriced else 'NO'}"
    )

# ==========================================
# 3. FRONT-END LAYOUT & AGENT CONFIG
# ==========================================
st.set_page_config(page_title="PropAgent AI", layout="wide")

instructions = """You are an elite Indian Real Estate Investment Advisor.
1. When someone asks about a price, use investment_advisor to check the deal.
2. Tell them clearly if they are getting a good deal (underpriced) or a bad deal.
3. Explicitly state the 'Potential to Rise' in Lakhs over the next 5 years.
4. Always report in LAKHS, never Crores."""

try:
   df_sample = pd.read_csv('housing_lite.csv')
except:
   st.sidebar.warning("Note: Dataset file not found, using pre-trained model.")

with st.sidebar:
    st.title("🏠 Price Predictor")
    sb_bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    sb_sqft = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1200)
    sb_year = st.number_input("Year Built", min_value=1950, max_value=2026, value=2020)
    
    if st.button("Quick Predict"):
        val = get_refined_prediction(sb_bhk, sb_sqft, sb_year)
        st.success(f"Price: ₹{val:.2f} Lakhs")
        
        st.markdown("---")
        st.subheader("📈 Price Trend")
        sizes = np.linspace(max(100, sb_sqft - 500), sb_sqft + 500, 10)
        preds = [get_refined_prediction(sb_bhk, s, sb_year) for s in sizes]
        df_plot = pd.DataFrame({"SqFt": sizes, "Price": preds})
        fig = px.line(df_plot, x="SqFt", y="Price", title="Price vs. Size")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. AI CHAT INTERFACE
# ==========================================
st.title("🤖 PropAgent AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready! Ask me about house prices or investment deals."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask: Is a 3BHK for 80L a good investment?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Using Gemini 2.5 Flash for 2026 stability
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            llm_with_tools = llm.bind_tools([property_price_predictor, investment_advisor])
            
            ai_msg = llm_with_tools.invoke(prompt)
            
            if ai_msg.tool_calls:
                responses = []
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "investment_advisor":
                        result = investment_advisor.invoke(tool_call["args"])
                    else:
                        result = property_price_predictor.invoke(tool_call["args"])
                    
                    # Passing tool output back to LLM for a natural response
                    final_resp = llm.invoke(f"{instructions}\n\nThe tool said: {result}. Explain this to the user.")
                    responses.append(final_resp.content) # Extracts clean text
                
                response_text = "\n\n".join(responses)
            else:
                response_text = ai_msg.content # Extracts clean text
                
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error(f"Chat Error: {e}")
