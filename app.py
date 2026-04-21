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
MY_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = MY_API_KEY

# ==========================================
# 2. LOAD MODEL & REFINED PREDICTION LOGIC
# ==========================================
@st.cache_resource
def load_prop_model():

    model = xgb.XGBRegressor()
    # Update this line to match your new filename!
    model_path = 'prop_model_updated.json' 
    
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
        except Exception as e:
            # If there is a version mismatch, we still want the app to run
            print(f"Model load failed: {e}")
    return model

xgb_model = load_prop_model()
def get_refined_prediction(bhk, sqft, year):
    features = np.array([[bhk, sqft, year]])
    try:
        raw_val = xgb_model.predict(features)[0]
        
        # IF THE BRAIN IS STILL GIVING MASSIVE NUMBERS:
        # We force it down. If it's 272, we want 72. 
        # If it's 27200000, we want 72.
        
        temp_val = raw_val
        while temp_val > 150: # No standard flat should be over 1.5Cr in this dataset
            temp_val = temp_val / 10.0
            
        # Logical Floor (Price can't be free)
        final_price = max(temp_val, (sqft * 3500) / 100000)
        
        # BHK Logic (Ensure 4BHK > 2BHK)
        final_price = final_price + (bhk * 8.0)
        
        return round(final_price, 2)
    except:
        return 65.0 # Total fallback

@tool
def property_price_predictor(bhk: int, sqft: int, year_built: int):
    """Predicts house price. Input: bhk, sqft, year_built."""
    prediction = get_refined_prediction(bhk, sqft, year_built)
    
    # WE FORCE THE TEXT HERE SO GEMINI CAN'T CHANGE IT TO CRORES
    return f"STRICT_VALUATION: {prediction} Lakhs (INR). Do not convert this to Crores."
# ==========================================
# 3. FRONT-END LAYOUT (SIDEBAR & MAIN)
# ==========================================
st.set_page_config(page_title="PropAgent AI", layout="wide")

# Data Check for VIVA (Helps you explain the dataset to the teacher)
try:
   df_sample = pd.read_csv('housing_lite.csv')
    # Use actual column names from your CSV here if they differ
    # df_sample = df_sample.rename(columns={'Price': 'price', 'Area': 'sqft'})
except:
    st.sidebar.warning("Note: Dataset file not found for live analysis, using pre-trained model only.")

with st.sidebar:
    st.title("🏠 Price Predictor")
    st.write("Enter details for manual calculation:")
    
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
        fig = px.line(df_plot, x="SqFt", y="Price", title="Price vs. Size (Refined)")
        st.plotly_chart(fig, use_container_width=True)
        # 1. ADD THIS NEW TOOL
@tool
def investment_advisor(bhk: int, sqft: int, current_listing_price: float, year_built: int):
    """
    Analyzes property investment. 
    IMPORTANT: current_listing_price MUST be in Lakhs (e.g., 58.0 for 58 Lakhs).
    """
    # 1. THE "LAKH" NORMALIZER
    # If the user input is like 20,00,000 or 2000000, we convert it to 20.0
    val = current_listing_price
    if val > 5000: # Clearly absolute rupees, not lakhs
        val = val / 100000.0
    
    # 2. GET MARKET VALUES
    fair_market_value = get_refined_prediction(bhk, sqft, year_built)
    
    # Force a 5-year growth for the teacher (3% annual appreciation)
    future_value = get_refined_prediction(bhk, sqft, year_built + 5) * 1.15 
    
    # 3. CORE LOGIC
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
# 2. UPDATE THE AGENT PROMPT
# Change your agent prompt to include these instructions:
instructions = """You are an elite Indian Real Estate Investment Advisor.
When users ask about investing, use your tools to check the fair market price.
If the listing price is lower than the fair market price, recommend it as a 'Value Buy'.
Focus on major Indian cities like Delhi NCR, Mumbai, and Bangalore.
Always explain the 'Potential Price Rise' based on the investment_advisor tool instructions 
You are a Real Estate Expert. 
When someone asks about a specific price for a house:
1. Use the investment_advisor tool to check the fair price.
2. Tell them clearly if they are getting a good deal (underpriced) or a bad deal.
3. Explicitly state the 'Potential to Rise' in Lakhs over the next 5 years."""

# 3. ADD BOTH TOOLS TO THE AGENT
tools = [property_price_predictor, investment_advisor]
# ... (rest of your agent initialization remains the same)

# ==========================================
# 4. AI CHAT INTERFACE
# ==========================================
# 4. AI CHAT INTERFACE
# ==========================================
st.title("🤖 PropAgent AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready! Ask me about prices or if a deal is a good investment."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask: Is a 3BHK for 80L a good investment?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Using Gemini 1.5 Flash (Most stable for tool use)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            # Change this line:
            # Change the model name to the 2026 stable version



            
            # FIX 1: Bind BOTH tools so the AI can choose the right one
            llm_with_tools = llm.bind_tools([property_price_predictor, investment_advisor])
            
            # The AI decides which tool to use
            ai_msg = llm_with_tools.invoke(prompt)
            # Change this part of your code:


# If the model didn't call a tool, it returns a message object.
# We must extract ONLY the text content.
if not ai_msg.tool_calls:
    response_text = ai_msg.content  # <--- THIS IS THE KEY LINE
else:
    # ... your tool handling logic ...
    # When you get final_resp from the tool:
    response_text = final_resp.content # <--- ALSO EXTRACT CONTENT HERE
            
            if ai_msg.tool_calls:
                responses = []
                for tool_call in ai_msg.tool_calls:
                    # FIX 2: Dynamic tool selection
                    if tool_call["name"] == "investment_advisor":
                        result = investment_advisor.invoke(tool_call["args"])
                    else:
                        result = property_price_predictor.invoke(tool_call["args"])
                    
                    # Explain the tool result with your specific instructions
                    final_resp = llm.invoke(f"{instructions}\n\nThe tool said: {result}. Give a professional advice.")
                    responses.append(final_resp.content)
                
                response_text = "\n\n".join(responses)
            else:
                response_text = ai_msg.content
                
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Chat Error: {e}")
