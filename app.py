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
# PASTE YOUR KEY INSIDE THE QUOTES BELOW
MY_API_KEY = "AIzaSyBC_Qs_xUJrMrf7E8mk4i4j84_mbEULGwA"
os.environ["GOOGLE_API_KEY"] = MY_API_KEY

# ==========================================
# 2. LOAD MODEL & DEFINE TOOL
# ==========================================
@st.cache_resource
def load_prop_model():
    model = xgb.XGBRegressor()
    # Make sure 'prop_model.json' is in the 'propagent' folder
    model.load_model('prop_model.json') 
    return model

xgb_model = load_prop_model()

@tool
def property_price_predictor(bhk: int, sqft: int, year_built: int):
    """Predicts house price. Input: bhk, sqft, year_built."""
    features = np.array([[bhk, sqft, year_built]])
    prediction = xgb_model.predict(features)[0]
    return f"The estimated market value is ₹{prediction:.2f} Lakhs."

# ==========================================
# 3. FRONT-END LAYOUT (SIDEBAR & MAIN)
# ==========================================
st.set_page_config(page_title="PropAgent AI", layout="wide")

with st.sidebar:
    st.title("🏠 Price Predictor")
    st.write("Enter details for manual calculation:")
    
    sb_bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    sb_sqft = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1200)
    sb_year = st.number_input("Year Built", min_value=1950, max_value=2026, value=2020)
    
    if st.button("Quick Predict"):
        # Single Prediction
        current_data = np.array([[sb_bhk, sb_sqft, sb_year]])
        val = xgb_model.predict(current_data)[0]
        st.success(f"Price: ₹{val:.2f} Lakhs")
        
        # Day 5 Graph
        st.markdown("---")
        st.subheader("📈 Price Trend")
        sizes = np.linspace(max(100, sb_sqft - 500), sb_sqft + 500, 10)
        preds = [xgb_model.predict(np.array([[sb_bhk, s, sb_year]]))[0] for s in sizes]
        
        df_plot = pd.DataFrame({"SqFt": sizes, "Price": preds})
        fig = px.line(df_plot, x="SqFt", y="Price", title="Price vs. Size")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. AI CHAT INTERFACE
# ==========================================
st.title("🤖 PropAgent AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready! Ask me about house prices."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask: What's the price of a 3BHK 1500sqft home?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            llm_with_tools = llm.bind_tools([property_price_predictor])
            
            ai_msg = llm_with_tools.invoke(prompt)
            
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    result = property_price_predictor.invoke(tool_call["args"])
                    final_resp = llm.invoke(f"The tool said: {result}. Explain this simply.")
                    response_text = final_resp.content
            else:
                response_text = ai_msg.content
                
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Chat Error: {e}")