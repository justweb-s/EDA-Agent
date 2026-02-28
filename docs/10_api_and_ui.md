# 10 — API and UI

## Backend: FastAPI

The backend is an async FastAPI application that exposes APIs for session management, event streaming, and HITL operations. It uses `asyncio` pervasively to avoid blocking the main thread during graph executions.

---

## API Endpoints

### Session Management

**`POST /sessions`**

Creates a new analysis session. It receives the file to analyze (multipart/form-data) and the session configuration.

Request body:
```
file: File (multipart)
provider: str          # openai, anthropic, ollama, groq
model: str             # the specific model
mode: str              # auto, chat
hitl_enabled: bool
user_instructions: str (optional)
```

Response:
```json
{
  "session_id": "uuid",
  "status": "created",
  "dataset_context_summary": {
    "file_name": "data.csv",
    "n_rows": 10000,
    "n_columns": 15,
    "detected_issues": []
  }
}
```

After the response, the backend starts the LangGraph run in a background async task. The frontend can immediately open the SSE stream to receive updates.

**`GET /sessions`**

Lists all sessions with their status. Supports filtering by status. Used by the UI for the initial screen and for the list of resumable sessions.

**`GET /sessions/{session_id}`**

Details for a specific session: status, current step, produced `n_cells`, timing, and any errors.

**`DELETE /sessions/{session_id}`**

Deletes a session and its checkpoint. Used to clean up old sessions.

---

### Streaming

**`GET /sessions/{session_id}/stream`**

The main SSE endpoint. Connects the client to the LangGraph event stream.

Headers response:
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no  (important for Nginx, disables buffering)
```

Each SSE event has the following form:
```
event: cell_added
data: {"cell_type": "code", "content": "...", "outputs": [...], "step_id": "dist_age"}

event: step_completed
data: {"step_id": "dist_age", "findings_summary": "..."}
```

If the client disconnects and reconnects, it can pass `?from_event=N` to receive only events after event N. This requires the backend to keep a buffer of the last N events per session.

---

### Human-in-the-Loop

**`GET /sessions/{session_id}/interrupt`**

Returns the current interrupt state if the session is "suspended". Includes the interrupt type, data to show the user, and available actions.

**`POST /sessions/{session_id}/resume`**

Resumes execution of a session suspended due to HITL.

Request body:
```json
{
  "action": "approve",
  "data": { ... }
}
```

The backend builds the appropriate `Command(resume=...)` and re-invokes the graph. The SSE stream resumes automatically.

---

### Download

**`GET /sessions/{session_id}/notebook`**

Downloads the produced `.ipynb` file. Available only for sessions with status `completed`.

**`GET /sessions/{session_id}/notebook/cells`**

Returns the current notebook cells as JSON. Used by the UI to rebuild the notebook preview without parsing the `.ipynb` file.

---

## Error Response Structure

All errors follow a standard structure:

```json
{
  "error": {
    "code": "STEP_FAILED",
    "message": "Step 'dist_age' failed after 3 retries",
    "details": {
      "step_id": "dist_age",
      "last_error": "KeyError: 'age'",
      "session_id": "uuid"
    },
    "recoverable": true,
    "suggested_action": "Try resuming the session or skipping this step"
  }
}
```

---

## Frontend: Chat UI

The UI is a single-page web application with three main areas that mirror the workflow.

### Area 1: Control panel (left)

Contains the file upload form and session configuration (LLM provider selection, mode, HITL), and the list of active and past sessions with their status.

Once the session starts, it shows the **progress tracker**: list of plan steps with status icons (pending, in_progress, completed, failed, skipped). Steps update in real time as SSE events arrive.

### Area 2: Notebook preview (center)

The main area where the notebook is built in real time. Each notebook cell appears as it is produced.

**Markdown cells** appear immediately with formatting (headings, formatted text).

**Code cells** appear in two phases: first code appears token-by-token as the CODER writes it (if token streaming is enabled), then outputs (text, charts) appear when the Executor finishes execution.

**Charts** are rendered inline in the notebook, exactly as in a real Jupyter Notebook.

A status bar below each cell shows its state: `generating`, `executing`, `completed`, `failed`.

### Area 3: Chat (right, visible only in chat mode)

The conversation area with the agent. Shows the conversation history and an input field for user messages.

When a `hitl_interrupt` event arrives, this area shows the approval panel with available options, temporarily overlaying the normal chat.

In single-shot mode, this area is not visible (or it shows only the system event log).

---

## Frontend State Management

The frontend maintains a local state synchronized with the backend via SSE events. The local state includes:

- The plan step list (for the progress tracker)
- The notebook cell list (for the preview)
- The current session status (in_progress, suspended, completed, failed)
- The current interrupt (for the HITL panel)

On reconnect (after an SSE disconnection), the frontend calls `GET /sessions/{session_id}` and `GET /sessions/{session_id}/notebook/cells` to rebuild local state, then reconnects to the SSE stream to receive new events.

---

## Authentication and Security

The system is designed for single-user usage (a developer tool), so it does not include complex authentication. In a basic deployment scenario:

All APIs are protected by a simple API key passed via the `X-API-Key` header. The key is configured via environment variable.

Uploaded files are stored in a temporary directory with UUID names to prevent collisions and path traversal. Files are automatically deleted after a session is completed or abandoned.

Code execution in the IPython kernel is already isolated by design (separate process), but for shared deployments you would evaluate using Docker containers for stronger isolation.
