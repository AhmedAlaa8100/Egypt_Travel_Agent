# Egypt Travel Agent

A modular, tool-using travel assistant built with Python, LangChain, and
LangGraph. The project is being developed incrementally, with a focus on
reliable tool execution, conversation continuity, retrieval-augmented
generation (RAG), persistent preference memory, and automated
evaluation.

> **Project status:** Sprint 1 --- reliable agent execution and
> LangGraph workflow. Sprint 2 will focus on a more complete
> Egypt-focused trip-planning workflow and trustworthy cost estimates.

## Table of contents

-   [Overview](#overview)
-   [Sprint 1 goals and capabilities](#sprint-1-goals-and-capabilities)
-   [Architecture](#architecture)
-   [Repository structure](#repository-structure)
-   [Current tools](#current-tools)
-   [Conversation state and memory](#conversation-state-and-memory)
-   [Setup](#setup)
-   [Configuration and secrets](#configuration-and-secrets)
-   [Running the tests](#running-the-tests)
-   [Optional live evaluation](#optional-live-evaluation)
-   [Current limitations](#current-limitations)
-   [Planned next steps](#planned-next-steps)
-   [Development principles](#development-principles)

## Overview

The goal is to build a practical travel-planning agent that can
understand a travel question, retrieve relevant information, call tools
when useful, handle tool failures, and maintain context across turns in
a conversation.

The current implementation is an early-stage prototype. It provides the
agent foundation and several travel-related tools; it is **not yet a
complete booking platform** and does not guarantee live prices,
availability, or reservations.

The initial product scope is Egypt-focused. The planned pricing approach
is hybrid:

-   Use live provider data when an accessible and appropriate API is
    available.
-   Use curated data or estimates when live information is unavailable.
-   Label the source, freshness, assumptions, and uncertainty of prices
    rather than presenting estimates as confirmed quotes.

## Sprint 1 goals and capabilities

Sprint 1 focused on making the agent's execution flow testable and more
reliable.

### Agent workflow

-   Uses a LangGraph state graph to coordinate model calls and tool
    execution.
-   Supports model responses that answer directly or request one or more
    tools.
-   Executes multiple tool calls from a model response and returns a
    corresponding `ToolMessage` for each call.
-   Routes tool results back to the model so it can continue reasoning
    and produce a final answer.
-   Enforces a configurable per-request model/tool iteration limit.
-   Handles unknown tool names, invalid tool inputs, retryable failures,
    and unexpected tool exceptions through the tool-execution layer.
-   Validates that incoming user messages and conversation thread IDs
    are non-empty strings.

### Conversation continuity

-   Uses LangGraph's `MessagesState` message reducer and a checkpointer.
-   Uses a `thread_id` to identify a conversation.
-   Sends only the new `HumanMessage` on each invocation; LangGraph
    restores checkpointed state for that thread.
-   Resets per-request counters and answer fields for each new user
    turn.
-   Keeps the system prompt outside the checkpointed conversation
    history and prepends it when calling the model.
-   Supports optional message trimming for the history sent to the
    model.

### Retrieval and memory

-   Includes a knowledge base for retrieving travel reference documents.
-   Includes preference tools for saving and recalling user preferences.
-   Keeps retrieval-augmented generation (RAG), long-term preference
    memory, and conversation checkpointing as separate concerns.

### Evaluation and tests

-   Includes unit and integration-style tests for the agent, tool
    execution, tool registry, preference tools, RAG, memory, and
    evaluation utilities.
-   Normal tests are designed to run with mocks or test doubles rather
    than requiring live model/API calls.
-   Live evaluation is guarded behind an explicit environment variable.

## Architecture

At a high level, a request follows this flow:

1.  The caller supplies a user message and a `thread_id`.
2.  The graph invokes the model with the system prompt and available
    conversation history.
3.  If the model returns a final answer, the graph ends.
4.  If the model requests tools, the graph checks the iteration limit.
5.  The tool node executes each requested tool and creates a matching
    `ToolMessage`.
6.  The graph invokes the model again with the tool results.
7.  The final answer is returned to the caller and the graph state is
    checkpointed.

``` text
User message + thread_id
          |
          v
      Model node
          |
          v
     Route response
       /       \
 final answer  tool calls
      |           |
      v           v
     END       Limit check
                 /    \
            limit hit  allowed
                |        |
                v        v
             Limit     Tools node
                |        |
                v        v
               END    Model node
```

### Main components

-   **LangGraph workflow:** coordinates model calls, tool execution,
    routing, and per-thread checkpoints.
-   **Chat model adapter:** constructs the configured chat model and
    binds the registered tools.
-   **Tool registry:** collects the tools available to the model.
-   **Tool execution layer:** executes tool calls, retries configured
    transient errors, and formats success/failure results.
-   **Knowledge base:** retrieves reference documents relevant to a
    travel question.
-   **Preference memory:** saves and recalls user preferences
    independently of the conversation's message history.
-   **Evaluation utilities:** support repeatable evaluation without
    requiring live calls during ordinary tests.

## Repository structure

The following is the intended high-level structure of the current
project. Some implementation details may evolve as later sprints are
completed.

``` text
travel_agent/
├── .env                     # Local secrets/configuration; do not commit
├── .gitignore
├── pytest.ini
├── requirements.txt
├── src/
│   └── travel_agent/
│       ├── agent.py
│       ├── config.py
│       ├── evaluation.py
│       ├── evaluation_runner.py
│       ├── llm/
│       │   └── model.py
│       ├── memory/
│       │   └── store.py
│       ├── rag/
│       │   └── store.py
│       └── tools/
│           ├── execution.py
│           ├── preference_tools.py
│           ├── registry.py
│           └── travel_tools.py
└── tests/
    ├── test_agent.py
    ├── test_agent_rag.py
    ├── test_knowledge_base.py
    ├── test_knowledge_base_integration.py
    ├── test_llm_model.py
    ├── test_memory_store.py
    ├── test_preference_tools.py
    ├── test_rag_tool.py
    ├── test_tool_binding.py
    ├── test_tool_execution.py
    ├── test_tool_registry.py
    └── test_evaluation.py
```

Test filenames and modules may change as the project develops; use the
repository's actual files as the source of truth.

## Current tools

The agent currently has these travel-related tool concepts:

  -----------------------------------------------------------------------
  Tool                                Purpose
  ----------------------------------- -----------------------------------
  `estimate_trip_cost`                Estimates hotel cost for a
                                      supported Egyptian destination
                                      using prototype nightly rates and
                                      an assumption of up to two
                                      travellers per room.

  `convert_currency`                  Converts between EGP, USD, and EUR
                                      using placeholder exchange rates.
                                      These rates are not live.

  `save_preference`                   Saves a user preference through the
                                      preference/memory layer.

  `recall_preferences`                Retrieves saved preferences
                                      relevant to a query.

  `search_knowledge_base`             Searches the travel knowledge base
                                      for relevant reference documents.
  -----------------------------------------------------------------------

### Important pricing limitation

The current `estimate_trip_cost` function estimates **hotel cost only**,
not the total cost of a trip. Its prototype nightly rates are hardcoded.
The current currency conversion rates are also placeholders, not live
market data.

Do not use these prototype values as confirmed travel quotes. A future
version is planned to use structured price records, explicit provenance,
cached/live exchange rates where available, and clearly labelled
estimates.

## Conversation state and memory

The project has three distinct kinds of state. They should not be
treated as interchangeable.

### 1. Conversation checkpointing

LangGraph's `InMemorySaver` stores graph checkpoints for each
`thread_id` in memory. It enables multi-turn continuity while the
process is running.

**Limitations:**

-   State is lost when the process restarts.
-   A thread ID is a conversation identifier, not an authentication or
    authorization mechanism.
-   A production service must authenticate users and verify that each
    user is allowed to access the requested thread.
-   Production deployment should use an appropriate persistent
    checkpointer and define retention and deletion policies.

### 2. Long-term preference memory

The `MemoryStore` is a separate component for saving and retrieving
preferences across conversations. It is not the same as LangGraph's
per-thread checkpoint.

### 3. RAG knowledge base

The knowledge base contains shared reference documents used to ground
answers. It is not private per-user conversation memory.

### Message trimming

The optional `max_history_tokens` setting controls the history passed to
the model. Trimming the model input does not automatically delete older
messages from the checkpointer's stored state.

## Setup

The commands below assume Windows PowerShell and the project directory:

``` text
C:\Users\DELL\Desktop\depi\travel_agent
```

Adjust the path if your repository is stored elsewhere.

### Requirements

-   Python 3.10 or a compatible Python version supported by the
    project's pinned dependencies.
-   PowerShell on Windows, or an equivalent shell.
-   Git (optional, for version control).
-   API credentials only if you intend to run the live model/evaluation
    path.

### 1. Open the project directory

``` powershell
cd C:\Users\DELL\Desktop\depi\travel_agent
```

### 2. Create a virtual environment if needed

If `.venv` already exists, you can skip environment creation.

``` powershell
py -3.10 -m venv .venv
```

Activate it:

``` powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation due to its execution policy, follow your
organization's or machine's approved Python environment guidance rather
than changing security policy blindly.

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check for dependency conflicts:

``` powershell
pip check
```

The project uses pinned dependencies to reduce unexpected
incompatibilities. Avoid upgrading the entire LangChain/LangGraph stack
without running the test suite afterward.

### 4. Configure Python imports

From the project root, set `PYTHONPATH` for the current PowerShell
session:

``` powershell
$env:PYTHONPATH = "src"
```

Set it again in each new terminal session before running commands that
import `travel_agent`, unless your development environment is configured
to install the project as a package.

### 5. Configure environment variables

Review `src/travel_agent/config.py` to see which environment variables
the current version expects. Create a local `.env` file if required by
that configuration and put secrets there.

Never commit API keys, tokens, or other secrets. Confirm that `.env` is
excluded by `.gitignore` before pushing the project to a remote
repository.

## Configuration and secrets

The agent reads configuration through `travel_agent.config`. The model
adapter is in `travel_agent.llm.model`.

Before running a live model call:

1.  Review `config.py` and `llm/model.py` to identify the exact settings
    required by the current implementation.
2.  Add the required credentials to your local environment or `.env`
    file.
3.  Keep credentials out of source code, logs, test fixtures,
    screenshots, and commits.
4.  Never print the contents of `.env` to debug configuration.
5.  Do not enable live evaluation unless you explicitly intend to make
    external requests.

The exact model/provider and settings can change as the project evolves;
consult those source files rather than assuming the README contains
every current environment variable.

## Running the tests

Run commands from the project root with the virtual environment
activated.

Set the import path:

``` powershell
$env:PYTHONPATH = "src"
```

Run the complete suite:

``` powershell
python -m pytest -v
```

Run a single test module:

``` powershell
python -m pytest tests/test_agent.py -v
```

Run one test by name:

``` powershell
python -m pytest tests/test_agent.py::test_agent_preserves_history_without_duplicate_messages -v
```

### Test expectations

-   Unit tests should not need a live model API.
-   Prefer fake models, mocks, and test doubles for model responses and
    tool calls.
-   Test both success and failure paths.
-   When testing repeated model calls, return fresh message objects with
    unique message IDs. LangGraph's message reducer can update an
    existing message when IDs match.
-   If a test fails, inspect the first relevant traceback and the actual
    messages/state passed through the graph before changing production
    code.

### Test categories

  -----------------------------------------------------------------------
  Area                                What it verifies
  ----------------------------------- -----------------------------------
  Agent workflow                      Direct answers, tool loops,
                                      iteration limits, and thread
                                      continuity

  Thread isolation                    One thread's checkpointed messages
                                      are not passed into another thread

  Tool execution                      Retry behavior, invalid arguments,
                                      unknown tools, and failures

  Tool registry and binding           Tools are registered and bound to
                                      the model as expected

  Preference memory                   Saving and recalling user
                                      preferences

  RAG                                 Retrieving relevant documents and
                                      passing tool results to the model

  Evaluation                          Evaluation data and runner
                                      behavior, including the
                                      live-evaluation guard
  -----------------------------------------------------------------------

## Optional live evaluation

Live evaluation can make external model/API calls. It is opt-in and
should not be run as part of the default offline test workflow.

In PowerShell:

``` powershell
$env:PYTHONPATH = "src"
$env:RUN_LIVE_EVAL = "1"
python -m travel_agent.evaluation_runner
Remove-Item Env:RUN_LIVE_EVAL
```

Only run this when the required provider credentials are configured and
you accept the possibility of API usage or charges. The final command
removes the opt-in variable from the current shell session.

If live evaluation fails, verify the provider configuration and API
access separately from the offline test suite.

## Current limitations

The Sprint 1 foundation is not yet a production travel service.

-   **Prototype prices:** accommodation rates and exchange rates are
    hardcoded placeholders.
-   **No guaranteed live inventory:** the agent does not yet guarantee
    real-time hotel, flight, train, or attraction availability.
-   **No booking execution:** it does not currently make reservations or
    purchases.
-   **In-memory checkpoints:** conversation state does not survive a
    process restart.
-   **Memory isolation:** production use needs user authentication,
    ownership checks, retention policies, and privacy safeguards around
    saved preferences.
-   **RAG quality:** the knowledge base is only as reliable and current
    as its source documents.
-   **Cost estimates:** the existing cost estimator calculates
    accommodation only; it does not yet produce a complete trip budget.
-   **Evaluation coverage:** passing tests verifies implemented
    behavior, not every possible real-world request or provider failure.
-   **Tool selection:** the current agent uses an LLM to decide when to
    call available tools. Future workflow work will reduce the number of
    model-facing tools and move arithmetic and pricing mechanics into
    deterministic internal services.

## Planned next steps

Sprint 2 is planned around an Egypt-focused trip-planning workflow.

1.  **Tool inventory and contracts:** inspect existing registration and
    preference tools before changing interfaces.
2.  **Structured data models:** define Pydantic models for trip
    requests, price results, cost entries, exchange-rate snapshots, tool
    errors, and budgets.
3.  **Thin end-to-end LangGraph workflow:** extract a structured trip
    request, ask for missing details, retrieve data, estimate costs, and
    return a budget using deterministic fixtures.
4.  **Price ranges and provenance:** support low/likely/high estimates,
    original currency, source, verification timestamp, and explicit
    incomplete-data status.
5.  **Egypt-specific cost rules:** account for applicable visitor
    categories, seasonality, mandatory charges, tipping, and entry/visa
    costs when reliable information is available.
6.  **Currency service:** verify EGP coverage and provider terms,
    implement bounded retries and caching, and define visible stale-rate
    behavior.
7.  **Selective live integrations:** add external providers only after
    verifying that access, endpoints, and data permissions are
    available.
8.  **Evaluation:** establish a 15--20-request golden evaluation set and
    track extraction, tool choice, calculation correctness,
    completeness, and provenance throughout development.

These items are planned work, not capabilities already implemented in
Sprint 1.

## Development principles

The project follows these principles as it grows:

-   **Deterministic math:** use ordinary Python for calculations instead
    of asking the LLM to do arithmetic.
-   **Typed contracts:** validate inputs and outputs with explicit
    schemas.
-   **Visible uncertainty:** distinguish live, cached, estimated, and
    unavailable information.
-   **Source provenance:** retain where a price came from and when it
    was last verified.
-   **Safe failures:** tool errors should be reported as errors, not
    successful results.
-   **No implicit purchases:** booking or payment actions must require
    explicit user approval.
-   **Offline testing:** normal tests should be repeatable without live
    provider calls.
-   **Incremental changes:** implement one feature at a time and run
    regression tests before moving forward.

------------------------------------------------------------------------

**Current milestone:** Sprint 1 complete --- agent execution,
conversation checkpointing, tool handling, RAG integration, and test
foundations.

**Next milestone:** a thin end-to-end Egypt trip planner with structured
requests, reliable cost aggregation, and explicit price provenance.
