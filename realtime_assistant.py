# realtime_assistant.py — GPT-4 assistant with latest OpenAI SDK

import json
import os
from datetime import datetime
import openai

# Load API key for new SDK
openai.api_key = os.getenv("OPENAI_API_KEY")

# Field dictionary for merchant form
form_data = {
    "SiteCompanyName1": None, "SiteAddress": None, "SiteCity": None, "SiteState": None, "SiteZip": None,
    "SiteVoice": None, "SiteFax": None, "CorporateCompanyName1": None, "CorporateAddress": None,
    "CorporateCity": None, "CorporateState": None, "CorporateZip": None, "CorporateName": None,
    "SiteEmail": None, "CorporateVoice": None, "CorporateFax": None, "BusinessWebsite": None,
    "CorporateEmail": None, "CustomerSvcEmail": None, "AppRetrievalMail": None, "AppRetrievalFax": None,
    "AppRetrievalFaxNumber": None, "MCC-Desc": None, "MerchantInitials1": None, "MerchantInitials2": None,
    "MerchantInitials3": None, "MerchantInitials4": None, "MerchantInitials5": None, "MerchantInitials6": None,
    "MerchantInitials7": None, "signer1signature1": None, "Owner0Name1": None, "Owner0LastName1": None,
    "signer1signature2": None, "Owner0Name2": None, "Owner0LastName2": None
}

conversation_history = []
last_assistant_msg = "Hello, can we get started by telling me the first steps?"
end_triggered = False

async def process_transcribed_text(user_text):
    global last_assistant_msg, end_triggered

    conversation_history.append({"role": "user", "text": user_text, "timestamp": datetime.now().isoformat()})

    # Step 1: Field extraction via GPT
    extraction_prompt = f"""
You are helping to fill out a merchant processing application form. Based on the assistant's question and the user's response, extract only the field values using the exact field names from this list:

{list(form_data.keys())}

Assistant: {last_assistant_msg}
User: {user_text}

Respond ONLY with a valid JSON object using only the field names above.
"""

    try:
        extract_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract structured fields from conversations."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.2
        )
        content = extract_response['choices'][0]['message']['content'].strip()
        if content.startswith("{") and content.endswith("}"):
            parsed = json.loads(content)
            for key, value in parsed.items():
                if key in form_data:
                    form_data[key] = value
    except Exception as e:
        print("⚠️ Field extraction error:", e)

    # Step 2: Continue assistant conversation
    instruction_prompt = """
You are an AI assistant designed to help users fill out a Merchant Processing Application and Agreement form. Greet the user in a friendly way when starting.
Your task is to guide the user through each section of the form, asking relevant questions to extract the necessary information required. 
Ensure that the conversation remains professional and user-friendly, providing explanations or examples when necessary to help the user understand the context of each question. DO NOT ANSWER ANY QUESTIONS NOT RELATED TO THE TASK AT HAND.
Always prioritize privacy and remind the user not to share sensitive information unless necessary for the form. For sections requiring specific types of data like percentages, business types, or legal requirements, 
offer examples to aid in understanding. Confirm each detail with the user before moving on to the next section. Only ask a couple of questions at a time and not all at once. 
Once all these fields are collected, read back the entire collected information to the user and ask them to confirm it and mention that it may take a few seconds to process all the information. 
After they confirm respond with 'END OF CONVERSATION' and nothing else.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()
        last_assistant_msg = assistant_reply
        conversation_history.append({"role": "assistant", "text": assistant_reply, "timestamp": datetime.now().isoformat()})

        if "END OF CONVERSATION" in assistant_reply.upper():
            end_triggered = True

        return assistant_reply

    except Exception as e:
        print("❌ Assistant generation failed:", e)
        return "Sorry, I had trouble processing that. Can you repeat?"
