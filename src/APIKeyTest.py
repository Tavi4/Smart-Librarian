import os
from openai import OpenAI
from dotenv import load_dotenv

from utils import load_openai_api_key


def main():

    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")

    if not key:
        key = load_openai_api_key(".env")
        os.environ["OPENAI_API_KEY"] = key

    print(" Using key prefix:", key[:10], "...")


    client = OpenAI(api_key=key)


    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a text about dragons."}
            ],
            max_tokens=50,
        )
        print(" Response:", resp.choices[0].message.content)
    except Exception as e:
        print(" API call failed:", type(e).__name__, "-", str(e))

if __name__ == "__main__":
    main()