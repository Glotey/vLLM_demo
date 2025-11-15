import time
from openai import OpenAI

# Configure the OpenAI client to connect to the local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # The API key can be a dummy value
)

# Define the model and the query
model_name = "gaunernst/gemma-3-12b-it-int4-awq"
query = "Why is the sky blue?"

print(f"Querying model: {model_name}")
print(f"Query: {query}\n")

# Record the start time
start_time = time.time()

# Send the request to the model
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": query}
        ],
        max_tokens=512,  # Set a reasonable max token limit for the response
        temperature=0.7,
    )

    # Record the end time
    end_time = time.time()

    # --- Performance Evaluation ---
    total_duration = end_time - start_time

    # Extract prompt and completion token counts from the response
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Calculate tokens per second (eval rate)
    if total_duration > 0:
        eval_rate = completion_tokens / total_duration
    else:
        eval_rate = float('inf')  # Avoid division by zero

    # --- Print Results ---
    generated_text = response.choices[0].message.content
    print("--- Response ---")
    print(generated_text)
    print("\n" + "="*30 + "\n")

    print("--- Evaluation Details ---")
    print(f"total duration:       {total_duration:.4f}s")
    print(f"prompt eval count:    {prompt_tokens} token(s)")
    print(f"eval count:           {completion_tokens} token(s)")
    print(f"eval rate:            {eval_rate:.2f} tokens/s")

except Exception as e:
    print(f"An error occurred: {e}")
