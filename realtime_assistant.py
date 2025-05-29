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

    # If all fields are filled, generate final summary from actual form_data
    if all_fields_filled:
        summary = generate_summary_from_form_data()
        end_triggered = True
        assistant_reply = (
            f"{summary}\n\nPlease confirm if all the details are correct. Once confirmed, it may take a few seconds to process."
        )
    else:
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

        except Exception as e:
            print("❌ Assistant generation error:", e)
            assistant_reply = "Sorry, I had trouble with that. Could you please repeat?"

    last_assistant_msg = assistant_reply
    conversation_history.append({
        "role": "assistant",
        "text": assistant_reply,
        "timestamp": datetime.now().isoformat()
    })

    if "END OF CONVERSATION" in assistant_reply.upper():
        end_triggered = True

    return assistant_reply


def generate_summary_from_form_data():
    def show(value):
        return value if value else "Not provided"

    summary = (
        f"Here is a summary of the information collected:\n\n"
        f" DBA Name: {show(form_data['DBAName'])}\n"
        f" Legal Corporate Name: {show(form_data['LegalCorporateName'])}\n"
        f" Business Address: {show(form_data['BusinessAddress'])}\n"
        f" Billing Address: {show(form_data['BillingAddress'])}\n"
        f" City: {show(form_data['City'])}\n"
        f" State: {show(form_data['State'])}\n"
        f" Zip: {show(form_data['Zip'])}\n"
        f" Phone: {show(form_data['Phone'])}\n"
        f" Fax: {show(form_data['Fax'])}\n"
        f" Contact Name: {show(form_data['ContactName'])}\n"
        f" Business Email: {show(form_data['BusinessEmail'])}\n"
        f" Contact Phone: {show(form_data['ContactPhone'])}\n"
        f" Contact Fax: {show(form_data['ContactFax'])}\n"
        f" Contact Email: {show(form_data['ContactEmail'])}\n"
        f" Website: {show(form_data['Website'])}\n"
        f" Customer Service Email: {show(form_data['CustomerServiceEmail'])}\n"
        f" Retrieval Request Destination: {show(form_data['RetrievalRequestDestination'])}\n"
        f" MCC SIC Description: {show(form_data['MCCSICDescription'])}"
    )
    return summary


def reset_assistant_state():
    global conversation_history, last_assistant_msg, end_triggered
    conversation_history.clear()
    last_assistant_msg = ""
    end_triggered = False
    for key in form_data:
        form_data[key] = None
