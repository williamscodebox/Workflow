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

        # --- Call LLM ---
        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
        )

        raw = response["message"]["content"].strip()

        # --- 1. Try direct JSON ---
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data
        except:
            pass

        # --- 2. Extract JSON array using robust regex ---
        # This matches ANYTHING between the first '[' and the last ']'
        match = re.search(r" \[. *] ", raw, flags=re.DOTALL)

        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list) and data:
                    return data
            except:
                pass

        # --- 3. Retry with stricter prompt ---
        retry = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": "Return ONLY a JSON array of 5 strings."},
                {"role": "user", "content": user_message.strip()},
            ],
        )

        retry_raw = retry["message"]["content"].strip()

        try:
            data = json.loads(retry_raw)
            if isinstance(data, list) and data:
                return data
        except:
            pass

        # --- 4. Final fallback (NEVER return empty) ---
        print("❌ LLM failed to return valid JSON twice.")
        print("Raw output:", raw)
        print("Retry output:", retry_raw)

        # crude but safe fallback
        return [content[:200]]
