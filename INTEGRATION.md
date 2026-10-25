# Connect Roamwise HTML to the existing TravelAgent

This is an integration overlay for the existing `travel_agent` project. It adds a FastAPI bridge and serves the UI from the same origin, so no CORS setup or browser-side API key is needed.

## Files in this overlay

- `src/travel_agent/api.py` — `/api/plan` endpoint calls the existing `TravelAgent.run(prompt, thread_id=...)`.
- `src/travel_agent/static/index.html` — the Roamwise UI wired to the endpoint.
- `requirements-additions.txt` — dependencies to add to your existing requirements.

## Install into your existing project

Copy the overlay's `src\travel_agent\api.py` and `src\travel_agent\static\index.html` into the matching locations in your existing project. Do not replace your existing `agent.py`, `.env`, or requirements file.

Append these two lines to the existing `requirements.txt`:

```text
fastapi
uvicorn[standard]
```

Then, in PowerShell:

```powershell
cd C:\Users\DELL\Desktop\depi\travel_agent
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]"
$env:PYTHONPATH = "src"
python -m uvicorn travel_agent.api:app --reload
```

Open **http://127.0.0.1:8000/**. Do not open the HTML by double-clicking it; the page must be served by FastAPI for `/api/plan` to work.

- Health check: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

The server initializes `TravelAgent` only when the first planning request arrives. Your existing `.env` remains local and is not sent to the browser. Because this page has no chat history UI, each submission uses a fresh `thread_id`, so an earlier plan cannot leak into a new one.

The browser sends only the form details to the backend. The backend validates them, calls the agent, repairs small JSON mistakes from the model, and returns the structured plan shown on the page. If the model returns an empty or unusable answer, the API retries up to three times.

## Cost estimates and fallback behavior

The page shows the agent's estimate with explicit warnings and never treats it as a live quote or a booking. It also shows a note when the model's stated total does not match its itemized costs.

`estimate_trip_cost` only supports Cairo, Alexandria, Luxor, and Aswan, and it estimates **accommodation only**. If every agent attempt fails to return usable cost figures, the API shows a clearly labelled baseline accommodation estimate for those cities. Meals, transport, flights, and activities remain excluded and are listed in the warnings. Unsupported destinations are left as "Not estimated" rather than given an invented number.

## Important limitation in the current agent

This connects the site to the current agent; it does not add provider integrations. Your current cost tool is not a complete verified whole-trip pricing engine. The prompt instructs the agent not to present a partial calculation as a complete trip total, and the UI labels every figure as an estimate.

If your local `agent.py` differs from the latest LangGraph version and does not accept `thread_id`, update the API call to match that method signature; the current project context indicates `run(user_message, thread_id)` is supported.
