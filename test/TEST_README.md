# ES-Chatbot Test Suite

This directory contains the pytest test suite used to validate the
ES-Chatbot system across all milestones.

## Test Structure
```
test/ 
  ├── test_m1_infrastructure.py 
  ├── test_m2_query_generation.py 
  ├── test_m3_safety.py
  ├── test_m4_chat_pipeline.py 
  ├── test_m5_frontend.py
  └── test_m6_hardening.py
```

| Test File | Milestone | Purpose |
|---|---|---|
| **test_m1_infrastructure.py** | Milestone 1 | Backend health and Elasticsearch connectivity |
| **test_m2_query_generation.py** | Milestone 2 | Natural language → Elasticsearch query generation |
| **test_m3_safety.py** | Milestone 3 | Query safety layer validation |
| **test_m4_chat_pipeline.py** | Milestone 4 | End‑to‑end chat pipeline |
| **test_m5_frontend.py** | Milestone 5 | Frontend behaviour validation |
| **test_m6_hardening.py** | Milestone 6 | Input validation and documentation checks |


## Prerequisites

Install test dependencies:

pip install -r pytest_requirements.txt

Run commands from the test directory.

------------------------------------------------------------------------

## Running Unit Tests (M2 / M3 / M5 / M6)

These tests run offline and do not require the backend (**Run commands from the Parent/Project directory**).

Windows CMD / PowerShell: pytest test\test_m2_query_generation.py test\test_m3_safety.py test\test_m5_frontend.py test\test_m6_hardening.py

Linux / macOS: pytest test/test_m2_query_generation.py
test/test_m3_safety.py test/test_m5_frontend.py
test/test_m6_hardening.py

------------------------------------------------------------------------

## Running Integration Tests (M1 / M4)

Integration tests require the backend server running.

Start backend example:

uvicorn backend.main:app --reload

or

docker compose up --build

Check backend health:

curl http://localhost:8000/health

Expected example output:
{"status":"ok","elasticsearch":true,"llm":true,"chromadb":true}

------------------------------------------------------------------------

## Run Integration Tests

Windows CMD: set RUN_INTEGRATION=1 && set
BACKEND_URL=http://localhost:8000&& pytest
test\test_m1_infrastructure.py
test\test_m4_chat_pipeline.py

PowerShell: \$env:RUN_INTEGRATION="1";
\$env:BACKEND_URL="http://localhost:8000"; pytest
test\test_m1_infrastructure.py
test\test_m4_chat_pipeline.py

Linux / macOS: RUN_INTEGRATION=1 BACKEND_URL=http://localhost:8000
pytest test/test_m1_infrastructure.py test/test_m4_chat_pipeline.py

------------------------------------------------------------------------

## Run All Tests

Windows CMD: set RUN_INTEGRATION=1 && set
BACKEND_URL=http://localhost:8000&& pytest test

PowerShell: \$env:RUN_INTEGRATION="1";
\$env:BACKEND_URL="http://localhost:8000"; pytest test

Linux / macOS: RUN_INTEGRATION=1 BACKEND_URL=http://localhost:8000
pytest test

------------------------------------------------------------------------

## Expected Results

### Without backend: 13 passed, 11 skipped
![Without backend running](.\image\wo_backend_acceptance_test_result.png)



### With backend running: 24 passed
![With backend running](.\image\w_backend_acceptance_test_result.png)

------------------------------------------------------------------------

## Troubleshooting

If integration tests fail, check backend health: curl
http://localhost:8000/health

Ensure BACKEND_URL has no trailing spaces.
