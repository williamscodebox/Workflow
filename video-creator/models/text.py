import json
import ollama
from typing import List
import re


class TextModel:
    def generate_content(self, topic: str="an historical inspiring fact that changed today's computer science world") -> str:

        system_message = """
        You are a YouTube Shorts Assistant help me to generate content. 
        No explanation, No additional text, No additional decorations, No screenplay, only content as plain paragraph but include commas, exclamations, question marks accordingly. 
        Use simple english, No roman representations.
        Ask for like, share and subscribe at the end of content.
        """

        user_message = f"""
        Create a 40-second content about {topic},

        Did you know...
        """

        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
        )

        content = response["message"]["content"]
        final_response = f"Welcome to the channel, {content}"

        return final_response

    def generate_image_prompts(self, content: str) -> List[str]:
        system_message = """
        You are an artist and a prompt engineer who describes images clearly.
        Respond ONLY with a JSON array of strings.
        No explanation. No extra text. No markdown. No prefix/suffix.
        """

        user_message = f"""
        Give list of 5 descriptive, clear image prompts for this content:

        {content}
        """

        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
        )

        raw = response["message"]["content"].strip()

        # --- 1. Fast path: valid JSON ---
        try:
            return list(json.loads(raw))
        except Exception:
            pass

        # --- 2. Attempt to extract JSON array from messy output ---
        try:
            # Extract anything between [ ... ]
            match = re.search(r" \[(.| \n) *\] ", raw)
            if match:
                cleaned = match.group(0)
            return list(json.loads(cleaned))
        except Exception:
            pass

        # --- 3. Retry once with a stricter system prompt ---
        retry_response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": "Return ONLY a JSON array of 5 strings."},
                {"role": "user", "content": user_message.strip()},
            ],
        )

        retry_raw = retry_response["message"]["content"].strip()

        try:
            return list(json.loads(retry_raw))
        except Exception:
            print("❌ LLM failed to return valid JSON twice.")
            print("Raw output:", raw)
            print("Retry output:", retry_raw)
            return []

