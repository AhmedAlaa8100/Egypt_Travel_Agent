# -*- coding: utf-8 -*-
"""

## Setup
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q openai chromadb sentence-transformers

import os, json, ast, operator, time
import chromadb
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        from google.colab import userdata
        GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    except Exception:
        pass
if not GROQ_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY")
    except Exception:
        pass
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found -- see Module 4 Session 1's lab intro for signup steps.")

from openai import OpenAI
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Setup complete.")

"""---
## Part A -- Complete Integration *(25%)*
"""

# A -- real memory (created ONCE, here)
memory_client = chromadb.Client()

try:
    memory_client.delete_collection("my_full_agent_memory")
except Exception:
    pass

memory_collection = memory_client.create_collection(name="my_full_agent_memory")

def remember(fact_text, metadata=None):
    embedding = embedder.encode([fact_text])[0].tolist()
    memory_collection.add(documents=[fact_text], embeddings=[embedding],
                           ids=[f"mem_{memory_collection.count()}"],
                           metadatas=[metadata or {"type": "agent_generated"}])

def recall(query, top_k=3):
    if memory_collection.count() == 0:
        return []
    query_embedding = embedder.encode([query]).tolist()
    results = memory_collection.query(query_embeddings=query_embedding, n_results=min(top_k, memory_collection.count()))
    return results["documents"][0]

# Prove remember() handles a call with no explicit metadata
remember("Alexandria is in Egypt")  # TODO: a real fact, no metadata argument
print("PASSED -- remember() safe with no explicit metadata.")

# A -- 3+ real tools
NIGHTLY_RATE_EGP = {"aswan": 2500, "cairo": 3000, "alexandria": 2800, "luxor": 2600}
FX_TO_EGP = {"EGP": 1.0, "USD": 48.0, "EUR": 52.0}   # placeholder rates

def estimate_trip_cost(destination, nights, travellers=1):
    """Hotel cost estimate in EGP for a destination."""
    dest = str(destination).strip().lower()
    if dest not in NIGHTLY_RATE_EGP:
        raise ValueError(f"Unknown destination '{destination}'. Choose from: {sorted(NIGHTLY_RATE_EGP)}")
    nights, travellers = int(nights), int(travellers)   # int("abc") raises ValueError by itself
    if not (1 <= nights <= 60) or not (1 <= travellers <= 10):
        raise ValueError("nights must be 1-60 and travellers 1-10")
    rooms = (travellers + 1) // 2                        # 2 people per room
    total = NIGHTLY_RATE_EGP[dest] * nights * rooms
    return {"destination": dest, "nights": nights, "rooms": rooms,
        "total_egp": total, "total_text": f"{total} EGP"}

def convert_currency(amount, from_currency, to_currency):
    """Convert between EGP, USD and EUR."""
    src, dst = str(from_currency).upper(), str(to_currency).upper()
    if src not in FX_TO_EGP or dst not in FX_TO_EGP:
        raise ValueError(f"Supported currencies: {sorted(FX_TO_EGP)}")
    amount = float(amount)                               # float("abc") raises ValueError by itself
    if amount < 0:
        raise ValueError("amount cannot be negative")
    result = amount * FX_TO_EGP[src] / FX_TO_EGP[dst]
    return {"amount": amount, "from": src, "to": dst, "result": round(result, 2)}

def save_preference(fact):
    """Store a traveller fact or preference in long-term memory."""
    if not isinstance(fact, str):
        raise TypeError("fact must be a string")
    if len(fact.strip()) < 3:
        raise ValueError("fact must be a non-empty sentence")
    remember(fact.strip(), {"type": "preference"})
    return {"saved": fact.strip()}

def recall_preferences(query):
    """Look up stored facts about the traveller."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must be a non-empty string")
    return {"memories": recall(query.strip(), top_k=3)}

TOOL_REGISTRY = {
    "estimate_trip_cost": estimate_trip_cost,
    "convert_currency": convert_currency,
    "save_preference": save_preference,
    "recall_preferences": recall_preferences,
}

tools_schema = [
    {"type": "function", "function": {
        "name": "estimate_trip_cost",
        "description": "Estimate the total hotel cost in EGP for a stay in an Egyptian city.",
        "parameters": {"type": "object", "properties": {
            "destination": {"type": "string", "description": "City: aswan, cairo, alexandria or luxor"},
            "nights": {"type": "integer", "description": "Number of nights (1-60)"},
            "travellers": {"type": "integer", "description": "Number of travellers (1-10), default 1"}},
            "required": ["destination", "nights"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between EGP, USD and EUR.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "EGP, USD or EUR"},
            "to_currency": {"type": "string", "description": "EGP, USD or EUR"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "save_preference",
        "description": "Save a fact or preference about the traveller (allergies, budget, dates) to long-term memory.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "A complete sentence stating the fact"}},
            "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "recall_preferences",
        "description": "Search long-term memory for stored facts about the traveller.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look up, e.g. 'food allergies'"}},
            "required": ["query"]}}},
]
print("Tools ready:", list(TOOL_REGISTRY.keys()))

SYSTEM_PROMPT = (
    "You are a travel assistant for Egypt. "
    "ALWAYS use the provided tools for hotel prices, currency conversion, "
    "destination advice (seasons, weather, crowds, cruises) and the traveller's saved preferences. "
    "Never answer these from your own knowledge, even if you think you know the answer. "
    "The 'Remembered' notes are only past chats and preferences; they are NOT reliable for prices or "
    "conversions, so still call the tools for those. "
    "Only give a final answer after you have the needed tool results in 'So far'. "
    "If a tool returns an error, explain it to the user instead of guessing."
    "Write numbers as plain digits with no thousands separators and no special spaces, "
    "followed by a normal space and the currency code, e.g. 33600 EGP, 646.15 EUR. "
)

def llm_think(goal, observation):
    memories_summary = "\n".join(f"- {m}" for m in observation["memories"]) or "(none yet)"
    history_summary = "\n".join(f"- {c['action']}: {c['result']}" for c in observation["context"])
    prompt = (f"Remembered (past chats and preferences only):\n{memories_summary}\n\n"
              f"Goal: {goal}\n\nSo far:\n{history_summary or '(nothing yet)'}\n\nNext step?")
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        tools=tools_schema,
    )
    message = response.choices[0].message
    if not message.tool_calls:
        return {"action": "terminate", "result": message.content}
    call = message.tool_calls[0]
    return {"action": call.function.name, "args": json.loads(call.function.arguments)}


def run_agent(goal, max_iterations=5, verbose=True):
    relevant_memories = recall(goal, top_k=3)
    context = []
    for i in range(max_iterations):
        observation = {"context": context, "memories": relevant_memories}
        decision = llm_think(goal, observation)
        if decision["action"] == "terminate":
            remember(f"Goal '{goal}' -> {decision['result']}")
            return {"status": "done", "result": decision["result"], "iterations": i}
        result, _ = execute_with_retry(TOOL_REGISTRY, decision["action"], decision.get("args", {}))
        if verbose:
            print(f"  [iter {i}] {decision['action']} -> {str(result)[:60]}")
        context.append({"action": decision["action"], "result": result})
    return {"status": "max_iterations_reached", "iterations": max_iterations}

"""---
## Part B -- Real Error Handling *(20%)*
"""

def execute_with_retry(tool_registry, action_name, args, max_retries=3):
    if action_name not in tool_registry:
        return f"Unknown tool: {action_name}", 0
    for attempt in range(max_retries):
        try:
            return tool_registry[action_name](**args), attempt
        except TimeoutError:
            if attempt == max_retries - 1:
                return f"Tool timed out after {max_retries} attempts", attempt
            time.sleep(2 ** attempt)
        except (ValueError, SyntaxError, TypeError) as e:
            # All genuinely mean "this input won't succeed on retry" -- fail fast.
            return f"Tool error (invalid input, not retried): {e}", attempt
    return "Unexpected retry exhaustion", max_retries

# Prove fail-fast against >=2 GENUINELY DIFFERENT kinds of bad input
# (e.g., a value error AND a type error/syntax error -- not two variations of the same one)
for name, bad_args in [("estimate_trip_cost", {"destination": "Paris", "nights": 3}), ("convert_currency", {"amount": -5, "from_currency": "USD", "to_currency": "EGP"}), ("estimate_trip_cost", {"destination": "aswan"})]:  # TODO: real bad-input cases
    result, attempts = execute_with_retry(TOOL_REGISTRY, name, bad_args, max_retries=3)
    print(f"{name}: {result} (attempts: {attempts + 1})")
    assert attempts == 0, "Expected fail-fast"  # TODO: uncomment once your tools raise real errors

"""---
## Part C -- RAG as a Tool *(15%)*
"""

doc_client = chromadb.Client()
try:
    doc_client.delete_collection("my_agent_docs")
except Exception:
    pass
doc_collection = doc_client.create_collection(name="my_agent_docs")

my_docs = ["Alexandria is your best destination in case you want a trip in winter", "Cairo is crowded and busy all year", "A Nile cruise between Luxor and Aswan usually takes 3 to 4 nights"]  # TODO: 2+ real documents
doc_collection.add(documents=my_docs, embeddings=embedder.encode(my_docs).tolist(),
                    ids=[f"doc{i}" for i in range(len(my_docs))])

def search_knowledge_base(query, top_k=1):
    results = doc_collection.query(query_embeddings=embedder.encode([query]).tolist(), n_results=top_k)
    return results["documents"][0] if results["documents"][0] else ["No matching document found."]

# VERIFY your test query actually finds the document BEFORE using it in Part D
print(search_knowledge_base("where to go in winter"))  # TODO: a real query -- check it actually matches!

TOOL_REGISTRY["search_knowledge_base"] = search_knowledge_base
# print(TOOL_REGISTRY)
tools_schema.append({"type": "function", "function": {"name": "search_knowledge_base",
    "description": "Search the knowledge base", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}})

"""---
## Part D -- Real Evaluation *(25%)*

**>=5 real scenarios.** Verify every retrieval-dependent query actually matches its target document first.
"""

import re
def evaluate_agent(agent_fn, test_scenarios):
    results = []
    for scenario in test_scenarios:
        t0 = time.time()
        outcome = agent_fn(scenario["goal"], verbose=False)
        elapsed = time.time() - t0
        success = outcome["status"] == "done" and scenario["check"](outcome.get("result", ""))
        results.append({"goal": scenario["goal"][:40], "success": success,
                         "iterations": outcome.get("iterations"), "time_s": round(elapsed, 2)})
    return results, sum(r["success"] for r in results) / len(results)

def norm(r):
    return re.sub(r"(?<=\d)[,\s\u00a0\u202f](?=\d)", "", str(r).lower())

my_test_scenarios = [
    # estimate_trip_cost: 1 room, 2500 * 3 nights = 7500
    {"goal": "How much will 3 nights in Aswan cost for 2 travellers?",
     "check": lambda r: "7500" in norm(r)},

    # estimate_trip_cost: 3 travellers need 2 rooms, 3000 * 4 * 2 = 24000
    {"goal": "Estimate a 4 night hotel stay in Cairo for 3 travellers.",
     "check": lambda r: "24000" in norm(r)},

    # convert_currency: 100 USD * 48 = 4800 EGP
    {"goal": "Convert 100 USD to EGP.",
     "check": lambda r: "4800" in norm(r)},

    # search_knowledge_base: winter advice
    {"goal": "Where should I go in Egypt for a warm winter trip?",
     "check": lambda r: "alexandria" in norm(r)},

    # search_knowledge_base: cruise length (3 to 4 nights)
    {"goal": "How many nights does a Nile cruise between Luxor and Aswan take?",
     "check": lambda r: "3" in norm(r) and "4" in norm(r)},

    # two tools chained: cost (2800 * 2 nights * 1 room = 5600 EGP), then USD (5600 / 48 = 116.67)
    {"goal": "How much is 2 nights in Alexandria for 2 travellers, and what is that in USD?",
     "check": lambda r: "5600" in norm(r) and ("116" in norm(r) or "117" in norm(r))}
]
results, success_rate = evaluate_agent(run_agent, my_test_scenarios)
for r in results:
    print(r)
print(f"\nSuccess rate: {success_rate:.0%}")



from pprint import pprint


pprint(run_agent("We are a family of 5 persons, we want to go to the perfect place in winter in egypt for 4 days, what is the cost needed in Euro for the whole trip, you choose everything without asking any questions and tell me your cost estimate", verbose=True))  # TODO: a real, moderately complex goal
