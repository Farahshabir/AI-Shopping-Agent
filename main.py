import os
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env")

# Set up Gemini client and model
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Product Search Function
def search_products(query: str) -> str:
    try:
        response = requests.get("https://hackathon-apis.vercel.app/api/products")
        response.raise_for_status()
        products = response.json()

        matches = [
            f"- {p['title']} | Rs {p['price']}"
            for p in products
            if query.lower() in p.get("title", "").lower()
        ]

        return "\n".join(matches[:5]) if matches else "❌ No matching products found."
    except Exception as e:
        return f"⚠️ API Error: {e}"

# Main App Function
def main():
    print("👋 Hello! 🛒 I'm your Shopping Assistant.")
    query = input("🔍 What would you like to search for today? 📝 ")

    agent = Agent(
        name="Shopping Agent",
        instructions="You help users find and recommend products.",
        model=model
    )

    # LLM-generated response
    answer = Runner.run_sync(agent, query, run_config=config).final_output

    # Product search results
    results = search_products(query)
    if results:
        print("\n🛍️ Products Found:\n" + results)

    print("\n🤖 Agent Response:\n" + answer)

# Entry Point
if __name__ == "__main__":
    main()
