"""
A minimal playground for the Gemini API.

Goal: one function, `call_gemini()`, that you can call with different
prompts and generation settings to build intuition for what each
parameter actually does — before wiring any of this into the FastAPI app.

Setup:
    pip install google-genai python-dotenv
    echo "GEMINI_API_KEY=your-key-here" > .env
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def call_gemini(
    prompt: str,
    model: str = "gemini-3.6-flash",
    temperature: float = 1.0,
    max_output_tokens: int = 500,
    top_p: float = 0.95,
    top_k: int = 40,
    system_instruction: str | None = None,
) -> str:
    """
    Send a prompt to Gemini and return the generated text.

    Parameters worth experimenting with:
        temperature: 0.0 = deterministic/repetitive, 1.0+ = more random/creative.
            Try the SAME prompt at 0.0 vs 1.5 a few times each and compare.
        top_p / top_k: alternate ways of controlling randomness by restricting
            which tokens the model is allowed to sample from. Usually left at
            defaults unless you're deliberately tuning output diversity.
        max_output_tokens: hard cap on response length. Set it low (e.g. 20)
            to see the response get cut off mid-sentence.
        system_instruction: sets the model's persona/behavior, separate from
            the user prompt itself. Try the same prompt with different system
            instructions to see how much it steers the response.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
        system_instruction=system_instruction,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    # --- transparency section: what the API actually gives you back ---
    print("=" * 60)
    print("RAW RESPONSE INFO")
    print("=" * 60)
    print(f"Model:          {model}")
    print(f"Finish reason:  {response.candidates[0].finish_reason}")
    if response.usage_metadata:
        print(f"Prompt tokens:      {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens:    {response.usage_metadata.candidates_token_count}")
        print(f"Total tokens:       {response.usage_metadata.total_token_count}")
    print("=" * 60)

    return response.text


if __name__ == "__main__":
    # Quick manual test — edit these and re-run to experiment.
    result = call_gemini(
        prompt="Explain what a transformer model is in two sentences.",
        temperature=0.7,
        max_output_tokens=100,
    )
    print("\nRESPONSE:\n")
    print(result)