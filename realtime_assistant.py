# realtime_assistant.py — updated to skip repeated greetings and force English replies

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
last_assistant_msg = "Hello, can we get started by telling me the first steps?"
end_triggered = False

async def process_transcribed_text(user_text):
    global last_assistant_msg, end_triggered

    conversation_history.append({"role": "user", "text": user_text, "timestamp": datetime.now().isoformat()})

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

Greet the user only once if necessary, then move through the form one section at a time. For each step, ask concise questions that gather business name, DBA, legal entity, address, phone, and all other relevant fields.
If the user says something unrelated or unclear, gently guide them back to the form-filling flow.
Once all required information is gathered, repeat the data back to the user and ask for confirmation.
Then say only: END OF CONVERSATION.
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

        # avoid repeating greeting every turn
        if last_assistant_msg.startswith("Hello") and len(conversation_history) > 1 and assistant_reply.startswith("Hello"):
            assistant_reply = "Thanks! Can you also tell me the business type (e.g., LLC, Corporation)?"

        last_assistant_msg = assistant_reply
        conversation_history.append({"role": "assistant", "text": assistant_reply, "timestamp": datetime.now().isoformat()})

        if "END OF CONVERSATION" in assistant_reply.upper():
            end_triggered = True

        return assistant_reply

    except Exception as e:
        print("❌ Assistant generation failed:", e)
        return "Sorry, I had trouble processing that. Can you repeat?"
