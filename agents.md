# Multi-Agent Pipeline - System Design Document

## 1. Project Overview

A research and report generation system using LangGraph for orchestration with multiple specialized agents. Designed to showcase clean architecture, extensibility, and optimization.

**Goal**: Demonstrate production-ready system design skills for recruiters.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                            │
│                   POST /research                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline Controller                         │
│         (Orchestrates agents, manages state, memory)          │
└──────┬──────────────────┬──────────────────┬───────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PLANNER    │◄──►│ RESEARCHER  │◄──►│   WRITER    │
│  Agent      │    │   Agent     │    │   Agent     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Memory Layer                               │
│   ┌─────────────────┐        ┌─────────────────┐              │
│   │  Session State  │        │ Knowledge Base  │              │
│   │  (In-memory)    │        │   (Qdrant)       │              │
│   └─────────────────┘        └─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Provider Layer (LLM)                         │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                   │
│   │  Ollama  │   │OpenRouter│   │  Groq    │  (with fallback) │
│   └──────────┘   └──────────┘   └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration Structure

### 3.1 Directory Structure

```
app/
├── config/
│   ├── __init__.py
│   ├── loader.py              # YAML config loader
│   ├── models.yaml            # Per-agent model + token settings
│   ├── agents.yaml            # Agent-specific instructions
│   └── system.yaml            # Global settings
├── agents/
│   ├── base.py               # Base agent class
│   ├── planner.py
│   ├── researcher.py
│   └── writer.py
├── protocol/
│   ├── __init__.py
│   ├── message.py            # Message schema
│   └── registry.py           # Agent registry
├── memory/
│   ├── __init__.py
│   ├── session.py            # Session state manager
│   └── knowledge.py          # Knowledge base interface
├── providers/
│   ├── __init__.py
│   ├── base.py               # Base provider
│   ├── ollama.py
│   ├── openrouter.py
│   └── factory.py            # Provider factory
├── pipeline/
│   └── pipeline.py           # LangGraph pipeline
└── main.py
```

### 3.2 models.yaml

```yaml
providers:
  primary: ollama
  fallback: openrouter

agents:
  planner:
    model: llama3.3
    provider: ollama
    max_output_tokens: 512
    temperature: 0.3
    reasoning_effort: null  # null = disabled
    
  researcher:
    model: llama3.3
    provider: ollama
    max_output_tokens: 1500
    temperature: 0.4
    reasoning_effort: null
    
  writer:
    model: llama3.3
    provider: ollama
    max_output_tokens: 2500
    temperature: 0.5
    reasoning_effort: null

token_limits:
  # Dynamic allocation based on depth
  shallow:
    planner: 256
    researcher: 512
    writer: 1024
  medium:
    planner: 512
    researcher: 1500
    writer: 2048
  deep:
    planner: 768
    researcher: 2500
    writer: 4000
```

### 3.3 agents.yaml

```yaml
base_instructions: |
  You are a helpful, accurate AI assistant.
  Always provide factual information.
  Cite your sources when possible.

agents:
  planner:
    role: Research Planner
    instructions: |
      You are a research planning expert.
      Your job is to break down a topic into 3-5 focused research questions.
      Questions should be specific, answerable, and cover different aspects.
      Format your response as a numbered list.
    output_format: numbered_list
    
  researcher:
    role: Researcher
    instructions: |
      You are a thorough researcher.
      Answer each research question with factual, detailed findings.
      Be objective and cite sources where possible.
      If unsure, indicate uncertainty.
    output_format: paragraphs
    
  writer:
    role: Technical Writer
    instructions: |
      You are a professional technical writer.
      Synthesize research findings into a clear, structured report.
      Include: Executive Summary, Key Findings (bullets), Detailed Analysis, Conclusion.
      Write in a professional, accessible tone.
    output_format: structured_report
```

### 3.4 system.yaml

```yaml
app:
  name: Multi-Agent Research Pipeline
  version: 2.0.0
  
memory:
  session:
    enabled: true
    max_history: 10
  knowledge_base:
    enabled: false  # Enable if you want persistent KB
    vector_db: qdrant
    collection: agent_memory
    
caching:
  enabled: true
  ttl_seconds: 3600
  provider: memory  # or redis
  
error_handling:
  max_retries: 2
  fallback_to_cache: true
```

---

## 4. Agent Specifications

### 4.1 Base Agent Interface

```python
class BaseAgent(Protocol):
    name: str
    role: str
    instructions: str
    model_config: ModelConfig
    
    async def run(self, state: AgentState) -> AgentState:
        ...
    
    async def validate_output(self, output: str) -> bool:
        ...
```

### 4.2 Agent State

```python
from typing import TypedDict, Optional
from datetime import datetime

class AgentState(TypedDict):
    # Pipeline state
    topic: str
    depth: str
    
    # Agent outputs
    plan: list[str]
    research_notes: list[str]
    report: str
    
    # Metadata
    agent_outputs: dict[str, str]  # Raw outputs for reference
    token_usage: dict[str, int]    # Track tokens per agent
    timestamps: dict[str, datetime]
    
    # Memory
    session_context: list[str]     # Accumulated context
    knowledge_base_hits: list[str] # Retrieved context
```

### 4.3 Agent Responsibilities

| Agent | Input | Output | Token Budget |
|-------|-------|--------|--------------|
| **Planner** | topic, depth, session_context | 3-5 research questions | 256-768 (dynamic) |
| **Researcher** | plan, knowledge_base | Factual findings per question | 512-2500 (dynamic) |
| **Writer** | topic, research_notes, session_context | Structured report | 1024-4000 (dynamic) |

---

## 5. Protocol Design

### 5.1 Agent Communication Protocol

```python
from pydantic import BaseModel
from enum import Enum

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    TOOL_CALL = "tool_call"

class AgentMessage(BaseModel):
    id: str
    sender: str
    receiver: str
    type: MessageType
    payload: dict
    timestamp: datetime
    metadata: dict = {}
```

### 5.2 Message Flow

```
┌──────────┐     Message      ┌──────────┐
│ Planner  │ ───────────────►│   LLM    │
│          │ ◄─────────────── │          │
└──────────┘   Response      └──────────┘
       │
       │ (state update)
       ▼
┌──────────┐     Message      ┌──────────┐
│Researcher│ ───────────────►│   LLM    │
│          │ ◄─────────────── │          │
└──────────┘   Response      └──────────┘
       │
       │ (state update)
       ▼
┌──────────┐     Message      ┌──────────┐
│  Writer  │ ───────────────►│   LLM    │
│          │ ◄─────────────── │          │
└──────────┘   Response      └──────────┘
```

### 5.3 Agent Registry

```python
class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        
    def register(self, agent: BaseAgent):
        self._agents[agent.name] = agent
        
    def get(self, name: str) -> BaseAgent:
        return self._agents[name]
        
    def list_agents(self) -> list[str]:
        return list(self._agents.keys())
```

---

## 6. Memory Strategy

### 6.1 Session Memory (Within Pipeline)

```python
class SessionMemory:
    def __init__(self, max_history: int = 10):
        self.context: list[str] = []
        self.max_history = max_history
        
    def add(self, agent_name: str, output: str):
        self.context.append(f"[{agent_name}]: {output[:200]}...")
        if len(self.context) > self.max_history:
            self.context = self.context[-self.max_history:]
            
    def get_context(self) -> str:
        return "\n".join(self.context)
    
    def clear(self):
        self.context = []
```

### 6.2 Knowledge Base (Optional - for future)

```
┌────────────────────────────────────┐
│         Knowledge Base             │
│  ┌────────────────────────────┐   │
│  │  Query: "AI trends"        │────┼──► Retrieved context
│  │  Collection: research      │    │     added to agent input
│  │  Top-K: 3                 │    │
│  └────────────────────────────┘   │
└────────────────────────────────────┘
```

---

## 7. Provider Layer

### 7.1 Factory Pattern

```python
class LLMProviderFactory:
    @staticmethod
    def create(provider: str, model: str, **kwargs) -> BaseLLM:
        providers = {
            "ollama": OllamaProvider,
            "openrouter": OpenRouterProvider,
            "groq": GroqProvider,
        }
        return providers[provider](model=model, **kwargs)
```

### 7.2 Fallback Logic

```python
async def call_with_fallback(prompt: str, primary: str, fallback: str):
    try:
        return await primary.invoke(prompt)
    except Exception as e:
        logger.warning(f"Primary failed: {e}, trying fallback")
        return await fallback.invoke(prompt)
```

---

## 8. Implementation Order

### Phase 1: Config System (Foundation)
1. Create config loader
2. Define YAML schemas
3. Implement environment variable overrides

### Phase 2: Provider Layer
1. Abstract base provider
2. Implement Ollama provider
3. Add OpenRouter as fallback
4. Create factory

### Phase 3: Agent Refactor
1. Create base agent class
2. Refactor planner/researcher/writer
3. Add token limit enforcement
4. Add custom instructions per agent

### Phase 4: Protocol & Memory
1. Implement message protocol
2. Create agent registry
3. Add session memory
4. (Optional) Knowledge base integration

### Phase 5: Pipeline
1. Refactor LangGraph pipeline
2. Add dynamic token allocation
3. Implement caching
4. Add error handling + retries

### Phase 6: Polish
1. Documentation (ARCHITECTURE.md, AGENT_SPEC.md)
2. Tests
3. Docker setup

---

## 9. Key Features for Recruiters

| Feature | What it Shows |
|---------|---------------|
| YAML-based config | DevOps thinking, maintainability |
| Provider fallback | Reliability engineering |
| Dynamic token limits | Cost optimization awareness |
| Session memory | State management skills |
| Protocol design | Clean interfaces |
| Type hints + Pydantic | Type safety |
| Modular architecture | Clean code principles |
| Tests | Quality assurance |

---

## 10. Files to Create/Modify

### New Files
- `app/config/loader.py`
- `app/config/models.yaml`
- `app/config/agents.yaml`
- `app/config/system.yaml`
- `app/agents/base.py`
- `app/protocol/message.py`
- `app/protocol/registry.py`
- `app/memory/session.py`
- `app/providers/base.py`
- `app/providers/ollama.py`
- `app/providers/factory.py`

### Modify
- `app/agents/planner.py` - Use config
- `app/agents/researcher.py` - Use config
- `app/agents/writer.py` - Use config
- `app/graph/pipeline.py` - Enhanced with memory
- `app/main.py` - Config initialization
