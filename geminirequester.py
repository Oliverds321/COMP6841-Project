import google.generativeai as genai
import time
from datetime import datetime
#done
API_KEY = "api key"
MODEL_NAME = "gemini-2.5-flash" #
TEMPERATURE = 1.5 #high for variety of answers
NUM_REQUESTS = 1
PROMPT = "prompt"

now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = current_time+"gemini_responses.txt"

def run_gemini_requests():
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"error connecting to api: {e}")
        print("apikey error")
        return

    model = genai.GenerativeModel(MODEL_NAME)
    generation_config = genai.GenerationConfig(
        temperature=TEMPERATURE
    )

    all_responses = []
    print(f"sending {NUM_REQUESTS} requests to model '{MODEL_NAME}'")

    for i in range(NUM_REQUESTS):
        print(f"sending request {i + 1}/{NUM_REQUESTS}")
        try:
            response = model.generate_content(
                PROMPT,
                generation_config=generation_config
            )
            all_responses.append(response.text)
            print("response received")
        except Exception as e:
            error_message = f"an error occurred on request {i + 1}: {e}"
            print(f"   <- {error_message}")
            all_responses.append(f"ERROR\n{error_message}\n")
        
        time.sleep(1)

    print(f"\nsaving {len(all_responses)} responses at {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for i, response_text in enumerate(all_responses):
                f.write(f" RESPONSE {i + 1} \n\n")
                f.write(response_text)
                if i < len(all_responses) - 1:
                    f.write("\n\n")
        print("Success")
    except IOError as e:
        print(f"error writing to file {e}")


if __name__ == "__main__":
    run_gemini_requests()