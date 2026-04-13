import argparse
import json
import os
import re
import time

from google import genai


def main():
    parser = argparse.ArgumentParser(description="Evaluate gene pairs using LLM.")
    parser.add_argument("--prompt_file", required=True, help="Path to the prompt markdown file.")
    parser.add_argument(
        "--pairs_file", required=True, help="Path to the JSON file containing gene pairs."
    )
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON.")
    args = parser.parse_args()

    with open(args.prompt_file, "r") as f:
        prompt_template = f.read()

    with open(args.pairs_file, "r") as f:
        pairs = json.load(f)

    formatted_pairs = "\n".join([f"- {pair}" for pair in pairs])
    prompt = prompt_template.replace("{pairs_list}", formatted_pairs)
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "").strip("\"'"))
    print(f"Querying LLM for {len(pairs)} pairs using prompt from {args.prompt_file}...")

    max_retries = 3
    result_data = {}

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview", contents=prompt
            )

            json_match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group(1))
            else:

                try:
                    result_data = json.loads(response.text)
                except Exception:
                    result_data = {"error": "No JSON found in response", "raw_text": response.text}
            break

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}")
                result_data = {"error": str(e)}
                break

            wait_time = (2**attempt) * 2  # Exponential backoff: 2s, 4s, 8s
            print(
                f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {wait_time}s... Error: {e}"
            )
            time.sleep(wait_time)

    # Save output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
