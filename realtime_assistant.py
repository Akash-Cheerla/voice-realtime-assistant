import json
import os
import random
from datetime import datetime
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

form_data = {
    "DBAName": None,
    "LegalCorporateName": None,
    "BusinessAddress": None,
    "BillingAddress": None,
    "City": None,
    "State": None,
    "Zip": None,
    "Phone": None,
    "Fax": None,
    "ContactName": None,
    "BusinessEmail": None,
    "ContactPhone": None,
    "ContactFax": None,
    "ContactEmail": None,
    "Website": None,
    "CustomerServiceEmail": None,
    "RetrievalRequestDestination": None,
    "MCCSICDescription": None
}

conversation_history = []
last_assistant_msg = ""
end_triggered = False


def get_initial_assistant_message():
    global last_assistant_msg
    greetings = [
        "Hi there! Ready to fill out your Merchant Application? Let's get started — what's your DBA or business name?",
        "Hello! I’ll be helping you fill out your merchant form. Let’s begin with your business’s DBA name.",
        "Welcome! Let’s kick things off. What’s the name your business operates under (DBA)?",
        "Hey! I’ll guide you through your Merchant form. First, can you tell me your DBA or business name?",
        "Great to have you! To start, what’s the doing-business-as (DBA) name for your company?"
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
You are helping fill out a Merchant Processing Application. Based on the assistant's last question and the user's reply, extract any relevant fields from this list:

{list(form_data.keys())}

Assistant: {last_assistant_msg}
User: {user_text}

Return only a valid JSON object using those exact field names. If nothing applies, return {{}}.
"""

    try:
        extract_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract structured form fields from user replies."},
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

    all_fields_filled = all(value is not None for value in form_data.values())

    instruction_prompt = """
You are a conversational AI assistant helping users fill out a Merchant Processing Application.

Be intelligent, friendly, and natural—like Siri or ChatGPT. Guide the user through collecting the following fields only:

- DBAName
- LegalCorporateName
- BusinessAddress
- BillingAddress
- City
- State
- Zip
- Phone
- Fax
- ContactName
- BusinessEmail
- ContactPhone
- ContactFax
- ContactEmail
- Website
- CustomerServiceEmail
- RetrievalRequestDestination
- MCCSICDescription

Ask one or two natural, context-aware questions at a time. Provide gentle examples if needed. Avoid robotic phrasing.

If the user says something irrelevant, gently steer them back.

Once all fields above are filled, summarize everything clearly and ask for confirmation. If the user confirms, say “END OF CONVERSATION” — no more questions after that.
"""

    try:
        messages = [
            {"role": "system", "content": instruction_prompt}
        ] + [
            {"role": msg["role"], "content": msg["text"]}
            for msg in conversation_history[-12:]
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
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
        return "Sorry, I had trouble with that. Could you please repeat?"
