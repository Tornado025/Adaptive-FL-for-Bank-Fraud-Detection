# Track 4 — Full-Stack Infrastructure & UI

**Role: The Platform Engineer**

You are the glue. Without your work, the other three tracks are isolated scripts that can never talk to each other. Your job is to build the network that connects them, the APIs that transport the model weights, and the dashboard that makes the entire system observable in real time.

You don't need to understand the math in Track 3. You don't need to understand the PyTorch model in Track 2. But you do need to understand every communication boundary — what data flows between containers, in what format, at what latency, and what happens when something goes wrong.

---

## File Structure

```
track4_infrastructure/
├── docker/
│   ├── Dockerfile.server          # Central server container (Track 3 + FastAPI)
│   ├── Dockerfile.bank_node       # Bank node container (Track 2 + SQLite DB)
│   ├── docker-compose.yml         # Orchestrates all containers + shared network
│   └── .env.example               # Environment variable template
├── api/
│   ├── main.py                    # FastAPI app entry point, lifespan management
│   ├── routes.py                  # All HTTP route handlers
│   ├── schemas.py                 # Pydantic models for request/response validation
│   ├── websocket_manager.py       # WebSocket connection pool and broadcast logic
│   └── grpc_transport.py          # gRPC client/stub for bank → server weight upload
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Dashboard.tsx      # Orchestrator view: global metrics, all banks
│       │   ├── BankClientView.tsx # Per-bank view: local loss, local AUC, status
│       │   ├── MetricsChart.tsx   # Recharts live chart component
│       │   └── RoundStatus.tsx    # Current FL round number and participant list
│       ├── pages/
│       │   ├── index.tsx          # Orchestrator dashboard (/)
│       │   └── bank.tsx           # Bank client view (/bank?id=bank_a)
│       └── hooks/
│           ├── useWebSocket.ts    # WebSocket connection hook
│           └── useMetrics.ts      # Metrics state management hook
├── tests/
│   ├── test_api.py                # FastAPI endpoint tests (pytest + httpx)
│   └── test_websocket.py          # WebSocket broadcast tests
└── requirements.txt
```

---

## Phase 1 — Docker Orchestration

**File: `docker/docker-compose.yml`**

Design a Docker Compose network with one server container and four bank node containers. All containers share a custom bridge network called `fl_network`.

**Container layout:**

```
fl_network (bridge)
├── central_server       (port 8000 exposed to host)
│   ├── Runs: FastAPI + Track 3 aggregation code
│   ├── Mounts: proxy_data/ volume
│   └── Image: Dockerfile.server
├── bank_node_a          (no external ports)
│   ├── Runs: Track 2 training loop script
│   ├── Mounts: bank_a.db volume (read-only)
│   └── Image: Dockerfile.bank_node
├── bank_node_b          (no external ports)
├── bank_node_c          (no external ports)
└── bank_node_d          (no external ports)
```

**`docker/Dockerfile.server`** requirements:
- Base: `python:3.11-slim`
- Copy `track3_federated_algorithms/` and `track4_infrastructure/api/`
- Install both `requirements.txt` files
- Expose port 8000
- CMD: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

**`docker/Dockerfile.bank_node`** requirements:
- Base: `python:3.11-slim`
- Copy `track2_deep_learning/`
- Install `requirements.txt`
- `ENV BANK_ID=bank_a` (overridden per-container in `docker-compose.yml`)
- `ENV SERVER_URL=http://central_server:8000`
- `ENV DB_PATH=/data/bank.db`
- CMD: `python src/train.py --federated`  (a flag that enables FL round mode vs. standalone mode)

**`docker/docker-compose.yml`** structure:
```yaml
services:
  central_server:
    build:
      context: ..
      dockerfile: docker/Dockerfile.server
    ports:
      - "8000:8000"
    networks:
      - fl_network
    volumes:
      - ../track3_federated_algorithms/proxy_data:/proxy_data:ro
    environment:
      - AGGREGATION_METHOD=custom
      - NUM_ROUNDS=20
      - MIN_CLIENTS_PER_ROUND=2

  bank_node_a:
    build:
      context: ..
      dockerfile: docker/Dockerfile.bank_node
    networks:
      - fl_network
    volumes:
      - ../track1_data_engineering/data/databases/bank_a.db:/data/bank.db:ro
    environment:
      - BANK_ID=bank_a
      - SERVER_URL=http://central_server:8000
    depends_on:
      - central_server

  # Repeat for bank_node_b, bank_node_c, bank_node_d with appropriate BANK_ID and db volume

networks:
  fl_network:
    driver: bridge
```

**Testing Phase 1:** Run `docker-compose up` and verify all 5 containers start, are on the same network, and can ping each other by container name.

---

## Phase 2 — API & Communications

**File: `api/schemas.py`**

Define all Pydantic models. These are the data contracts. Do not skip validation — malformed weight packages from a buggy Track 2 implementation must fail loudly, not silently corrupt the global model.

```python
class WeightPayload(BaseModel):
    bank_id: str
    round: int
    num_samples: int
    weights: dict[str, list]  # Key: layer name, Value: nested list (tensor as JSON)
    metadata: dict            # val_loss, val_auc, local_epochs_trained

class AggregationResponse(BaseModel):
    round: int
    global_weights: dict[str, list]
    diagnostics: dict         # reliability scores, conflict penalties per bank

class RoundStatus(BaseModel):
    current_round: int
    total_rounds: int
    participating_banks: list[str]
    status: str               # "waiting", "aggregating", "broadcasting", "complete"

class MetricsUpdate(BaseModel):              # WebSocket broadcast payload
    event: str                               # "round_complete", "client_update", etc.
    round: int
    bank_id: str | None
    metrics: dict                            # auc, loss, f1, accuracy
```

**File: `api/routes.py`**

```python
# POST /upload_weights
# Called by each bank node after local training. Receives a WeightPayload.
# When all expected banks have uploaded for the current round,
# triggers aggregation (calls Track 3) and broadcasts the result.
@router.post("/upload_weights")
async def upload_weights(payload: WeightPayload): ...

# GET /global_weights/{round_number}
# Called by bank nodes at the start of each round to download the latest global weights.
@router.get("/global_weights/{round_number}")
async def get_global_weights(round_number: int): ...

# GET /status
# Returns the current FL round status (for polling by the dashboard as a fallback).
@router.get("/status")
async def get_status() -> RoundStatus: ...

# GET /history
# Returns the full history of metrics across all completed rounds.
@router.get("/history")
async def get_history() -> list[dict]: ...

# WebSocket /ws
# Clients (browsers) connect here to receive live metric updates.
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): ...
```

**Bank-to-server weight transport:** Bank nodes send weights via HTTP POST to `/upload_weights`. The payload can be large (hundreds of MB for larger models), so implement streaming uploads:

```python
# In train.py (Track 2 - bank side), called after extract_base_weights()
import httpx

async def send_weights_to_server(weight_package: dict, server_url: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{server_url}/upload_weights",
            json=weight_package,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()  # Returns the new global weights
```

Provide this function to Track 2 as a code snippet they integrate into `train.py`.

---

## Phase 3 — Next.js Dashboard Foundation

**Initialize the frontend:**
```bash
cd track4_infrastructure
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend && npm install recharts lucide-react
```

**Page routing:**

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | `Dashboard` | Orchestrator view — global model metrics, all banks at a glance |
| `/bank?id=bank_a` | `BankClientView` | Individual bank view — local training metrics, connection status |

**File: `frontend/src/components/Dashboard.tsx`**

The main orchestrator view displays:
- Current FL round number (large, prominent)
- Global model AUC-ROC (live updating line chart, last 20 rounds)
- Per-bank reliability score (bar chart, updated after each round)
- Conflict penalty heatmap (4x4 grid of cosine similarities from `round_diagnostics`)
- List of banks currently participating in this round with their status indicators

**File: `frontend/src/components/BankClientView.tsx`**

Each bank's individual view displays:
- Bank ID and profile description
- Local training loss (live updating per epoch during training)
- Local AUC-ROC (val set, per round)
- Connection status to the server (green/red)
- Last N round contributions (were their weights accepted? what was their reliability score?)

**File: `frontend/src/components/MetricsChart.tsx`**

A reusable Recharts wrapper. Must support:
- Line chart mode (training metrics over rounds/epochs)
- Bar chart mode (per-bank comparisons)
- Real-time data append without full re-render (use `useState` + `useRef` for the data array)
- Configurable Y-axis domain (0–1 for AUC, auto for loss)
- Dark mode support (follow system preference)

---

## Phase 4 — Real-Time Telemetry

**File: `api/websocket_manager.py`**

Manage WebSocket connections from the Next.js frontend. Multiple browser tabs may connect simultaneously.

```python
class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a MetricsUpdate to all connected browser clients."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                dead_connections.append(connection)
        for conn in dead_connections:
            self.disconnect(conn)
```

**Events to broadcast** (emit these from `routes.py` at the appropriate moments):

| Event | When | Payload |
|-------|------|---------|
| `client_update_received` | Bank POST arrives | bank_id, round, val_auc |
| `aggregation_started` | All banks uploaded | round, participating_banks |
| `round_complete` | Aggregation done | round, global_auc, diagnostics |
| `training_epoch` | Per-epoch (bank pushes this) | bank_id, epoch, train_loss, val_auc |

**File: `frontend/src/hooks/useWebSocket.ts`**

```typescript
export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<MetricsUpdate | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(url);
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect after 3 seconds
      setTimeout(() => { /* reconnect */ }, 3000);
    };
    ws.current.onmessage = (event) => {
      setLastMessage(JSON.parse(event.data));
    };
    return () => ws.current?.close();
  }, [url]);

  return { isConnected, lastMessage };
}
```

**File: `frontend/src/hooks/useMetrics.ts`**

```typescript
export function useMetrics() {
  const [roundHistory, setRoundHistory] = useState<RoundMetrics[]>([]);
  const [bankMetrics, setBankMetrics] = useState<Record<string, BankMetrics>>({});
  const { lastMessage } = useWebSocket(`ws://localhost:8000/ws`);

  useEffect(() => {
    if (!lastMessage) return;
    if (lastMessage.event === "round_complete") {
      setRoundHistory(prev => [...prev, {
        round: lastMessage.round,
        globalAuc: lastMessage.metrics.global_auc,
        diagnostics: lastMessage.metrics.diagnostics
      }]);
    }
    if (lastMessage.event === "training_epoch" && lastMessage.bank_id) {
      setBankMetrics(prev => ({
        ...prev,
        [lastMessage.bank_id!]: { ...lastMessage.metrics }
      }));
    }
  }, [lastMessage]);

  return { roundHistory, bankMetrics };
}
```

---

## Integration Checklist

**Before go-live, verify:**

- [ ] `docker-compose up` starts all 5 containers with no errors
- [ ] Bank nodes can reach server by hostname: `curl http://central_server:8000/status` from inside a bank container
- [ ] POST to `/upload_weights` with a sample payload returns 200 and a valid `AggregationResponse`
- [ ] WebSocket connection from browser to `ws://localhost:8000/ws` succeeds
- [ ] Dashboard at `http://localhost:3000` loads and shows "Waiting for round 1..."
- [ ] After a full simulated round, the live chart updates in the browser within 2 seconds of round completion
- [ ] Graceful handling: if a bank container crashes mid-round, the server waits for a configurable timeout then proceeds with available banks

**Environment variables (`.env.example`):**
```env
# Server config
AGGREGATION_METHOD=custom        # fedavg or custom
NUM_ROUNDS=20
MIN_CLIENTS_PER_ROUND=2          # Proceed if at least 2 banks upload
ROUND_TIMEOUT_SECONDS=300        # Wait up to 5 min for all banks

# Bank node config (set per container)
BANK_ID=bank_a
SERVER_URL=http://central_server:8000
DB_PATH=/data/bank.db
LOCAL_EPOCHS=3
BATCH_SIZE=256
```

---

## Dependencies

**Python (FastAPI server):**
```
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.0
httpx>=0.26
websockets>=12.0
python-multipart>=0.0.9
```

**Node.js (Next.js frontend):**
```
next>=14.0
react>=18
recharts>=2.10
lucide-react>=0.300
typescript>=5.0
tailwindcss>=3.4
```

Install: `pip install -r requirements.txt` and `cd frontend && npm install`
