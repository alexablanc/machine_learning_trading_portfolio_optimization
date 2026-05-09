import os
import openai

def get_llm_signal(bb_pb, rsi_val, momentum_val):
    """
    Use OpenAI API to generate a directional trading signal based on indicators.
    Returns: 0 (Bearish/Short), 1 (Neutral/Cash), 2 (Bullish/Long)
    """
    # Use the pre-configured OPENAI_API_KEY environment variable
    client = openai.OpenAI()
    
    system_prompt = """You are a quantitative trading assistant. Your task is to analyze technical indicators and output a single integer representing your trading signal:
    0 = Bearish (Expect price to drop, recommend shorting)
    1 = Neutral (Unclear direction, recommend holding cash)
    2 = Bullish (Expect price to rise, recommend going long)
    
    You must reply with ONLY a single integer (0, 1, or 2). Do not include any other text or explanation.
    """
    
    user_prompt = f"""Current indicator values:
    Bollinger %B (20-day): {bb_pb:.4f} (Values < 0.2 suggest oversold, > 0.8 suggest overbought)
    RSI (14-day): {rsi_val:.2f} (Values < 40 suggest oversold, > 60 suggest overbought)
    Momentum (10-day): {momentum_val:.4f} (Positive = uptrend, Negative = downtrend)
    
    Based on mean-reversion and momentum confluence, what is your signal?"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        
        signal_text = response.choices[0].message.content.strip()
        signal = int(signal_text)
        
        # Ensure it's a valid action
        if signal not in [0, 1, 2]:
            print(f"Warning: LLM returned invalid signal '{signal_text}', defaulting to Neutral (1)")
            return 1
            
        return signal
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Default to Neutral on error
        return 1
