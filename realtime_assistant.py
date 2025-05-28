# realtime_assistant.py — updated with get_initial_assistant_message and refined flow

import json
import os
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


def get_initial_assistant_message():
    global last_assistant_msg
    initial_message = (
        "Hello! I'm here to assist you in filling out your Merchant Processing Application and Agreement form. "
        "Let's start with the first section. \n\nCould you please provide the legal name of your business?"
    )
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
Always respond in English, regardless of what language the user speaks. Do NOT repeat introductory greetings once the conversation has started.

Move through the form one section at a time. Ask concise, friendly questions to collect each required field (business name, DBA, entity type, address, phone, website, emails, initials, etc).

If the user goes off-topic or speaks unclearly, gently guide them back.

Once everything is filled, summarize the data and say only: END OF CONVERSATION.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.3
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        if last_assistant_msg.lower().startswith("hello") and len(conversation_history) > 1 and assistant_reply.lower().startswith("hello"):
            assistant_reply = "Thanks! Can you also tell me the business type (e.g., LLC, Corporation)?"

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
        return "Sorry, I had trouble processing that. Can you repeat?"
