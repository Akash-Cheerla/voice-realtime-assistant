import json
import os
import random
from datetime import datetime
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

form_data = {
    "SiteCompanyName1": None,
    "CorporateCompanyName1": None,
    "SiteAddress": None,
    "CorporateAddress": None,
    "SiteCity": None,
    "SiteState": None,
    "SiteZip": None,
    "SiteVoice": None,
    "SiteFax": None,
    "CorporateVoice": None,
    "CorporateFax": None,
    "SiteEmail": None,
    "CorporateEmail": None,
    "BusinessWebsite": None,
    "CustomerSvcEmail": None,
    "AppRetrievalMail": None,
    "AppRetrievalFax": None,
    "AppRetrievalFaxNumber": None,
    "MCC-Desc": None,
    "MerchantInitials1": None,
    "signer1signature1": None,
}

conversation_history = []
last_assistant_msg = ""
end_triggered = False
summary_given = False


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
            label = field.replace("SiteCompanyName1", "DBA Name").replace("CorporateCompanyName1", "Legal Corporate Name")\
                .replace("SiteAddress", "Business Address").replace("CorporateAddress", "Billing Address")\
                .replace("SiteCity", "City").replace("SiteState", "State").replace("SiteZip", "Zip")\
                .replace("SiteVoice", "Phone").replace("SiteFax", "Fax").replace("CorporateVoice", "Contact Phone")\
                .replace("CorporateFax", "Contact Fax").replace("SiteEmail", "Business Email")\
                .replace("CorporateEmail", "Contact Email").replace("BusinessWebsite", "Website")\
                .replace("CustomerSvcEmail", "Customer Service Email")\
                .replace("AppRetrievalMail", "Send Retrieval Requests to Business Address")\
                .replace("AppRetrievalFax", "Send Retrieval Requests to Fax")\
                .replace("AppRetrievalFaxNumber", "Retrieval Request Fax Number")\
                .replace("MCC-Desc", "MCC SIC Description")\
                .replace("MerchantInitials1", "Initials").replace("signer1signature1", "Merchant Signature")
            summary_lines.append(f"{label}: {value}")
    return "\n\nHere is a summary of the information collected:\n\n" + "\n".join(summary_lines) + \
        "\n\nPlease confirm if all the details are correct. Once confirmed, it may take a few seconds to process."


async def process_transcribed_text(user_text):
    global last_assistant_msg, end_triggered, summary_given

    conversation_history.append({
        "role": "user",
        "text": user_text,
        "timestamp": datetime.now().isoformat()
    })

    # Field extraction
    extraction_prompt = f"""
You are helping fill out a Merchant Processing Application. Extract only relevant fields from this list:

{list(form_data.keys())}

Assistant: {last_assistant_msg}
User: {user_text}

Return a valid JSON with any matched fields. Use booleans for checkboxes like AppRetrievalMail or AppRetrievalFax.
If nothing matches, return {{}}
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

    all_fields_filled = all(form_data[k] is not None for k in [
        "SiteCompanyName1", "CorporateCompanyName1", "SiteAddress", "SiteCity", "SiteState", "SiteZip",
        "SiteVoice", "CorporateVoice", "SiteEmail", "CorporateEmail", "BusinessWebsite", "CustomerSvcEmail",
        "AppRetrievalMail", "AppRetrievalFax", "MCC-Desc"
    ])

    # Give summary only once
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

    # End conversation if confirmed
    if summary_given and any(w in user_text.lower() for w in ["yes", "confirmed", "correct", "that's right"]):
        end_triggered = True
        final_msg = "END OF CONVERSATION"
        last_assistant_msg = final_msg
        conversation_history.append({
            "role": "assistant",
            "text": final_msg,
            "timestamp": datetime.now().isoformat()
        })
        return final_msg

    # Continue assistant flow
    instruction_prompt = """
You are a friendly AI assistant helping the user fill out a Merchant Application. Only ask for the following:

- DBA Name
- Legal Corporate Name
- Business Address
- Billing Address (if different)
- City, State, Zip
- Phone / Fax / Contact Name / Contact Phone / Contact Email
- Business Email
- Website
- Customer Service Email
- Send Retrieval Requests: AppRetrievalMail (business address), AppRetrievalFax (fax) as booleans
- Retrieval Request Fax Number
- MCC/SIC Description
- Initials and Merchant Signature

Only ask 1–2 questions at a time. When everything is filled, summarize exactly once. Wait for confirmation. Then say "END OF CONVERSATION" and stop.

DO NOT repeat the summary. DO NOT say END OF CONVERSATION more than once.
"""

    try:
        messages = [{"role": "system", "content": instruction_prompt}]
        messages += [{"role": m["role"], "content": m["text"]} for m in conversation_history[-12:]]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        # Avoid repeated ending
        if summary_given and ("summary" in assistant_reply.lower() or "END OF CONVERSATION" in assistant_reply.upper()):
            return ""

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
    global conversation_history, last_assistant_msg, end_triggered, summary_given
    conversation_history.clear()
    last_assistant_msg = ""
    end_triggered = False
    summary_given = False
    for key in form_data:
        form_data[key] = None
