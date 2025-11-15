import time
import threading
from openai import OpenAI

# --- Configuration ---
# Make sure this matches the model your server is running
MODEL_NAME = "gaunernst/gemma-3-12b-it-int4-awq"
PROMPT = "Write a short story about a robot who discovers music."
MAX_TOKENS = 512
NUM_CONCURRENT_REQUESTS = 10  # Number of users sending requests at the same time

# --- OpenAI Client Setup ---
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# --- Global variables to store results ---
total_completion_tokens = 0
total_time_taken = 0
lock = threading.Lock()

def send_request(request_id):
    global total_completion_tokens
    global total_time_taken

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": PROMPT}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        end_time = time.time()
        duration = end_time - start_time
        completion_tokens = response.usage.completion_tokens

        # Use a lock to safely update shared variables
        with lock:
            total_completion_tokens += completion_tokens
            # We will use the total time for all threads, not individual durations
        
        print(f"Request {request_id}: Finished in {duration:.2f}s, Generated {completion_tokens} tokens.")

    except Exception as e:
        print(f"Request {request_id}: An error occurred: {e}")

# --- Main execution ---
print(f"Starting benchmark with {NUM_CONCURRENT_REQUESTS} concurrent requests...")
threads = []
benchmark_start_time = time.time()

for i in range(NUM_CONCURRENT_REQUESTS):
    thread = threading.Thread(target=send_request, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

benchmark_end_time = time.time()
total_time_taken = benchmark_end_time - benchmark_start_time

# --- Calculate and Print Results ---
if total_time_taken > 0:
    # This is the TRUE throughput number
    throughput = total_completion_tokens / total_time_taken
else:
    throughput = float('inf')

print("\n" + "="*30)
print("--- Benchmark Results ---")
print(f"Total time to complete all {NUM_CONCURRENT_REQUESTS} requests: {total_time_taken:.2f}s")
print(f"Total tokens generated: {total_completion_tokens}")
print(f"Overall Throughput: {throughput:.2f} tokens/s")
print("="*30)
