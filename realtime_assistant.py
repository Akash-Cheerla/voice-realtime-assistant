# realtime_assistant.py — enhanced with variable greetings and structured conversational flow

import json
import os
import random
from datetime import datetime
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

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
last_assistant_msg = ""
end_triggered = False

initial_greetings = [
    "Hi there! Let’s get started on your Merchant Processing Application.",
    "Hello! I'm here to guide you through your Merchant Application form. Shall we begin?",
    "Welcome! I’ll be assisting you with your Merchant Processing details today.",
    "Hey! Let’s work together to complete your Merchant Application step-by-step."
]


def get_initial_assistant_message():
    global last_assistant_msg
    initial_message = random.choice(initial_greetings) + "\n\nTo start, could you please provide the legal name of your business?"
    last_assistant_msg = initial_message
    conversation_history.append({
        "role": "assistant",
        "text": initial_message,
        "timestamp": datetime.now().isoformat()
    })
    return initial_message


async def process_transcribed_text(user_text):
    global last_assistant_msg, end_triggered

    conversation_history.append({
        "role": "user",
        "text": user_text,
        "timestamp": datetime.now().isoformat()
    })

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

    instruction_prompt = """
You are an AI assistant designed to help users fill out a Merchant Processing Application and Agreement form.
Greet the user in a friendly way only once at the start. Do not repeat the greeting in future replies.

Your task is to guide the user through each section of the form, asking relevant questions to extract the necessary information required.
Ensure the tone is intelligent, friendly, and conversational — like Siri or ChatGPT. Keep questions short and helpful.
DO NOT answer questions not related to the form.

Ask only one or two related questions at a time. For example, first ask for the business name, then the DBA, then address details, etc.
For business types, legal fields, or MCC codes, give examples when needed.

At the end, read back all collected details in a clear summary and say:
“Thanks for confirming! It might take a few seconds to process your form...” and then respond ONLY with 'END OF CONVERSATION'.
"""

    try:
        assistant_input = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in conversation_history[-6:]])

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": assistant_input}
            ],
            temperature=0.3
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        last_assistant_msg = assistant_reply
        conversation_history.append({
            "role": "assistant",
            "text": assistant_reply,
            "timestamp": datetime.now().isoformat()
        })

        if "END OF CONVERSATION" in assistant_reply.upper():
            end_triggered = True

        return assistant_reply

    except Exception as e:
        print("❌ Assistant generation failed:", e)
        return "Sorry, I had trouble processing that. Can you say it again?"
