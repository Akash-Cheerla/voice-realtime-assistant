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
summary_given = False
summary_confirmed = False


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


def build_summary_from_form():
    summary_lines = []
    for field, value in form_data.items():
        if value:
            field_name = field.replace("MCCSICDescription", "MCC SIC Description").replace("DBAName", "DBA Name")
            field_name = field_name.replace("LegalCorporateName", "Legal Corporate Name").replace("BusinessAddress", "Business Address")
            field_name = field_name.replace("BillingAddress", "Billing Address").replace("CustomerServiceEmail", "Customer Service Email")
            field_name = field_name.replace("RetrievalRequestDestination", "Retrieval Request Destination")
            field_name = ' '.join([w.capitalize() for w in field_name.split()])
            summary_lines.append(f"{field_name}: {value}")
    summary = "\n\nHere is a summary of the information collected:\n\n" + "\n".join(summary_lines)
    summary += "\n\nPlease confirm if all the details are correct. Once confirmed, it may take a few seconds to process."
    return summary


async def process_transcribed_text(user_text):
    global last_assistant_msg, end_triggered, summary_given, summary_confirmed

    conversation_history.append({
        "role": "user",
        "text": user_text,
        "timestamp": datetime.now().isoformat()
    })

    # Extract structured fields from user response
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

    # Show summary only once
    if all_fields_filled and not summary_given:
        summary_given = True
        summary = build_summary_from_form()
        last_assistant_msg = summary
        conversation_history.append({
            "role": "assistant",
            "text": summary,
            "timestamp": datetime.now().isoformat()
        })
        return summary

    # Check for confirmation after summary
    if summary_given and not summary_confirmed:
        if any(phrase in user_text.lower() for phrase in ["yes", "correct", "confirmed", "looks good", "all good"]):
            summary_confirmed = True
            end_triggered = True
            final_msg = "END OF CONVERSATION"
            last_assistant_msg = final_msg
            conversation_history.append({
                "role": "assistant",
                "text": final_msg,
                "timestamp": datetime.now().isoformat()
            })
            return final_msg

    # Otherwise, keep asking remaining questions
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
Always prioritize privacy and remind the user not to share sensitive information unless necessary for the form. For sections requiring specific types of data like percentages, business types, or legal requirements, 
offer examples to aid in understanding.Once all these fields are collected, read back the entire collected information to the user and ask them to confirm it and mention that it may take a few seconds to process all the information .
 After they confirm respond with 'END OF CONVERSATION' and nothing else.

DO NOT REPEAT THE SUMMARY. DO NOT REPEAT END OF CONVERSATION.
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

        if summary_given and ("summary" in assistant_reply.lower() or "end of conversation" in assistant_reply.lower()):
            return ""  # avoid repetition

        last_assistant_msg = assistant_reply
        conversation_history.append({
            "role": "assistant",
            "text": assistant_reply,
            "timestamp": datetime.now().isoformat()
        })

        return assistant_reply

    except Exception as e:
        print("❌ Assistant generation error:", e)
        return "Sorry, I had trouble with that. Could you please repeat?"


def reset_assistant_state():
    global conversation_history, last_assistant_msg, end_triggered, summary_given, summary_confirmed
    conversation_history.clear()
    last_assistant_msg = ""
    end_triggered = False
    summary_given = False
    summary_confirmed = False
    for key in form_data:
        form_data[key] = None
