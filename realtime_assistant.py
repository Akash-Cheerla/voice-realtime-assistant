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

def get_initial_assistant_message():
    global last_assistant_msg
    greetings = [
        "Hi there! Ready to fill out your Merchant Application? Let's get started — what's the legal name of your business?",
        "Hello! I'm your assistant for this form. Can you please tell me your business’s legal name?",
        "Welcome! Let's kick off the form. What's the official name of your business entity?",
        "Hey! I’ll guide you through your Merchant Application. First up: your legal business name, please?",
        "Great to have you here! To begin, could you share your business's legal name?"
    ]
    initial_message = random.choice(greetings)
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
You are helping fill out a merchant processing application. Based on the assistant's last question and the user's reply, extract any fields from this form:

{list(form_data.keys())}

Assistant: {last_assistant_msg}
User: {user_text}

Return a valid JSON object using only those field names. If nothing matches, return an empty JSON object like {{}}.
"""

    try:
        extract_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract structured fields from the user's speech."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.2
        )
        extracted_json = extract_response['choices'][0]['message']['content'].strip()
        parsed = json.loads(extracted_json)
        for key, value in parsed.items():
            if key in form_data:
                form_data[key] = value
    except Exception as e:
        print("⚠️ Field extraction error:", e)

    instruction_prompt = """
You are an AI assistant designed to help users fill out a Merchant Processing Application and Agreement form.
Be friendly, conversational, and intelligent like Siri or ChatGPT. Your goal is to collect all necessary fields from the user.

Required fields include: business name (DBA), legal name, address, city, state, zip, phone, fax, website, business email, customer service email, contact person, MCC/SIC code, initials, and signatures.

DO NOT repeat greetings once the conversation has started. Ask only a couple of questions at a time, and use natural, friendly wording. Offer simple examples when needed (e.g., “For example, www.mybusiness.com”).

If the user says something off-topic, gently bring them back to the form.

Once all the required fields are collected, summarize all the form data nicely, ask the user to confirm everything, and mention it may take a few seconds to process.

If they confirm, end the conversation with: END OF CONVERSATION — nothing else.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction_prompt},
                *[
                    {"role": msg["role"], "content": msg["text"]}
                    for msg in conversation_history[-10:]
                ]
            ],
            temperature=0.4
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
        print("❌ Assistant generation error:", e)
        return "Sorry, I had trouble processing that. Could you please repeat?"
