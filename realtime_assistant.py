import json
import os
import random
from datetime import datetime
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Form field structure
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

    # Save user input
    conversation_history.append({
        "role": "user",
        "text": user_text,
        "timestamp": datetime.now().isoformat()
    })

    # Prompt to extract form fields
    extraction_prompt = f"""
You are helping fill out a merchant processing application form.
Based on the assistant's last question and the user's reply, extract any of the following fields:

{json.dumps(list(form_data.keys()), indent=2)}

Assistant: {last_assistant_msg}
User: {user_text}

Return ONLY a valid JSON object with matched fields and values.
If there are no matches, return just {{}}.
"""

    try:
        extract_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract structured fields from the user's speech."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0
        )
        extracted_json = extract_response['choices'][0]['message']['content'].strip()
        parsed = json.loads(extracted_json)
        for key, value in parsed.items():
            if key in form_data:
                form_data[key] = value
    except Exception as e:
        print("⚠️ Field extraction error:", e)

    # Main assistant logic prompt
    instruction_prompt = """
You are an AI assistant designed to help users fill out a Merchant Processing Application and Agreement form.
Be friendly, intelligent, and conversational — like Siri or ChatGPT.

Your job is to collect all necessary form fields from the user in a natural way.
NEVER repeat greetings after the first message. DO NOT answer questions unrelated to the form.

Ask only 1–2 questions at a time. Be helpful. Provide examples when needed.
If the user goes off-topic, gently steer them back.

Once all required fields are collected, summarize the completed form in a friendly bullet list.
Ask the user to confirm all the details. If they confirm, reply ONLY with:

END OF CONVERSATION
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction_prompt},
                *[
                    {"role": msg["role"], "content": msg["text"]}
                    for msg in conversation_history[-12:]
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
        return "Sorry, I had trouble processing that. Could you please say it again?"
