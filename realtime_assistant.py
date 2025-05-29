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
summary_shown = False


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


def generate_final_summary():
    summary = []
    field_labels = {
        "DBAName": "1. DBA Name",
        "LegalCorporateName": "2. Legal Corporate Name",
        "BusinessAddress": "3. Business Address",
        "BillingAddress": "4. Billing Address",
        "City": "5. City",
        "State": "6. State",
        "Zip": "7. Zip",
        "Phone": "8. Phone",
        "Fax": "9. Fax",
        "ContactName": "10. Contact Name",
        "BusinessEmail": "11. Business Email",
        "ContactPhone": "12. Contact Phone",
        "ContactFax": "13. Contact Fax",
        "ContactEmail": "14. Contact Email",
        "Website": "15. Website",
        "CustomerServiceEmail": "16. Customer Service Email",
        "RetrievalRequestDestination": "17. Retrieval Request Destination",
        "MCCSICDescription": "18. MCC SIC Description"
    }
    for key, label in field_labels.items():
        value = form_data.get(key)
        if value:
            summary.append(f"{label}: {value}")
    return "\n".join(summary)


async def process_transcribed_text(user_text):
    global last_assistant_msg, end_triggered, summary_shown

    conversation_history.append({
        "role": "user",
        "text": user_text,
        "timestamp": datetime.now().isoformat()
    })

    if summary_shown and user_text.lower() in ["yes", "correct", "everything is correct", "that's right", "confirmed"]:
        end_triggered = True
        return (
            "🟢 Excellent! Thank you for confirming. Your Merchant Processing Application is now ready "
            "for submission with all the provided details. If you have any more questions or need further assistance "
            "in the future, feel free to ask. Have a great day! END OF CONVERSATION"
        )

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

    if all_fields_filled and not summary_shown:
        summary_shown = True
        return (
            "Thank you for providing all the necessary details. Here's a summary:\n\n"
            f"{generate_final_summary()}\n\n"
            "✅ Please confirm if all the details are correct. Once confirmed, it may take a few seconds to process."
        )

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
        messages = [{"role": "system", "content": instruction_prompt}]
        messages += [{"role": msg["role"], "content": msg["text"]} for msg in conversation_history[-12:]]

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


def reset_assistant_state():
    global conversation_history, last_assistant_msg, end_triggered, summary_shown
    conversation_history.clear()
    last_assistant_msg = ""
    end_triggered = False
    summary_shown = False
    for key in form_data:
        form_data[key] = None
