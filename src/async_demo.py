import asyncio
import time

# SYNC version - slow
def sync_fake_api_call(name):
    time.sleep(2)  # simulates waiting for LLM response
    return f"Response from {name}"

# ASYNC version - fast
async def async_fake_api_call(name):
    await asyncio.sleep(2)  # non-blocking wait
    return f"Response from {name}"

# Run 3 sync calls
def run_sync():
    start = time.time()
    results = []
    results.append(sync_fake_api_call("API Call 1"))
    results.append(sync_fake_api_call("API Call 2"))
    results.append(sync_fake_api_call("API Call 3"))
    print(f"Sync took: {time.time() - start:.1f}s")
    print(results)

# Run 3 async calls simultaneously
async def run_async():
    start = time.time()
    results = await asyncio.gather(
        async_fake_api_call("API Call 1"),
        async_fake_api_call("API Call 2"),
        async_fake_api_call("API Call 3"),
    )
    print(f"Async took: {time.time() - start:.1f}s")
    print(list(results))

print("--- SYNC ---")
run_sync()

print("\n--- ASYNC ---")
asyncio.run(run_async())