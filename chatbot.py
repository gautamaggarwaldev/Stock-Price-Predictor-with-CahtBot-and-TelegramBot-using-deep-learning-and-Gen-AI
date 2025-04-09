from flask import Flask, request, jsonify
import google.generativeai as genai  # type: ignore
import os
from datetime import datetime
import logging
import time
from flask_cors import CORS
from google.generativeai.types import HarmCategory, HarmBlockThreshold

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini with retry mechanism
def initialize_gemini(max_retries=3, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY environment variable not set")
                return None
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini model initialized successfully")
            return model
        except Exception as e:
            retries += 1
            logger.error(f"Attempt {retries} failed to initialize Gemini: {str(e)}")
            if retries < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    logger.error(f"Failed to initialize Gemini after {max_retries} attempts")
    return None

# DoraFinance system prompt
SYSTEM_PROMPT = """You are DoraFinance, an expert AI stock market advisor with 50+ years of experience in financial markets.
Your role is to provide insightful, accurate, and responsible guidance on stock market questions.

Guidelines:
1. Provide clear, concise explanations suitable for both beginners and experienced investors
2. When discussing specific stocks or investment strategies, ALWAYS include a disclaimer that this is not financial advice
3. Include relevant metrics when analyzing companies (P/E ratio, market cap, revenue growth, debt-to-equity, etc.)
4. Explain technical terms in accessible language, adding brief definitions for specialized terminology
5. For price predictions, discuss multiple factors that could influence the stock's movement rather than giving specific price targets
6. Consistently emphasize the importance of diversification, risk management, and investing with a long-term horizon
7. When information is limited or outdated, acknowledge limitations and avoid speculation
8. Suggest reliable sources for further research when appropriate
9. Consider macroeconomic factors and industry trends in your analysis
10. Be mindful of cognitive biases that affect investment decisions
11. Also explain the potential risks and rewards of any investment strategy you discuss
12. Avoid making absolute statements; instead, use phrases like "may", "could", or "might" to indicate uncertainty
13. If a question is outside your expertise, politely decline to answer and suggest consulting a financial advisor
15. Explain market technical terms and concepts in a way that is easy to understand for someone without a finance background
16. Avoid overly technical jargon unless necessary, and provide definitions when you do use it
17. Be aware of the potential for market manipulation and the ethical implications of your responses
18. Avoid discussing or promoting any illegal activities related to the stock market
19. Be cautious about discussing specific stocks or investment strategies that could be considered insider trading or market manipulation
20. If user ask any financial question or query please provide a solution not telling any illogical advice and knowledge.
21. At last also add a some random useful tips about share market and stock market investment in every answer.

Remember: Your guidance could influence financial decisions. Be thorough, balanced, and responsible in your responses.
"""

safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE}
]

# Initialize the model
model = initialize_gemini()

@app.route('/ask', methods=['POST'])
def ask_question():
    global model
    
    if not model:
        model = initialize_gemini()
        if not model:
            return jsonify({'error': 'AI model not available'}), 503
        
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        current_date = datetime.now().strftime("%B %d, %Y")
        full_prompt = SYSTEM_PROMPT.replace("{current_date}", current_date)
        full_prompt += f"\n\nQuestion: {question}"
        
        response = model.generate_content(full_prompt, safety_settings=safety_settings)
        
        if not hasattr(response, 'text'):
            logger.error("Response from Gemini has no 'text' attribute")
            return jsonify({'error': 'Invalid response from AI model'}), 500
        
        return jsonify({
            'question': question,
            'answer': response.text,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': 'Failed to process question', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host='0.0.0.0', port=port)