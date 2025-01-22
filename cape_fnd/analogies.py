import openai
from config import OPENAI_API_KEY, OPENAI_API_BASE

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

def get_veracity_analogies(article):
    prompt = f"""
    Given the article: "{article}", please recall similar cases and provide analogies where unverified natural remedies were claimed to cure serious diseases.
    Describe the situations and the outcomes.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()