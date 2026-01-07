# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system for querying course materials. It uses ChromaDB for vector storage, Anthropic's Claude API with tool use for intelligent responses, and provides a FastAPI backend with a web frontend.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Set up environment (required)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running the Application
```bash
# Quick start (from project root)
./run.sh

# Manual start (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points:
# - Web UI: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

### Working with ChromaDB
```bash
# Clear and rebuild vector database (requires modifying code)
# In Python: vector_store.clear_all_data()
# Then restart server to reload documents from docs/ folder

# Database location: ./backend/chroma_db/
```

## Architecture

### Request Flow (User Query → Response)

1. **Frontend** → POST `/api/query` → **app.py** (FastAPI endpoint)
2. **app.py:66** → `rag_system.query()` → **rag_system.py**
3. **rag_system.py:102** → Retrieves conversation history from `SessionManager`
4. **rag_system.py:122** → Calls `ai_generator.generate_response()` with tool definitions
5. **ai_generator.py:80** → First Claude API call (Claude decides if it needs to search)
6. **If Claude uses tool** → `ai_generator._handle_tool_execution()`:
   - **search_tools.py:52** → `CourseSearchTool.execute()`
   - **vector_store.py:61** → `VectorStore.search()`
   - If `course_name` provided → **vector_store.py:102** → Semantic search in `course_catalog` to resolve exact title
   - **vector_store.py:93** → Semantic search in `course_content` with filters
   - ChromaDB performs cosine similarity search using Sentence Transformers embeddings
   - **search_tools.py:88** → Format results with course/lesson context
7. **ai_generator.py:134** → Second Claude API call with tool results
8. **rag_system.py:137** → Save conversation to session history
9. **rag_system.py:140** → Return (answer, sources) to API
10. **app.py:68** → Return JSON response to frontend

### Core Components

**RAGSystem** (`rag_system.py`) - Main orchestrator
- Coordinates all components
- Manages query flow and session handling
- Single entry point: `query(query, session_id)`

**VectorStore** (`vector_store.py`) - ChromaDB management
- **Two collections**:
  - `course_catalog`: Course metadata (title, instructor, lessons) - used for resolving course names semantically
  - `course_content`: Actual content chunks with filters (course_title, lesson_number)
- `search()` method handles both course name resolution and content search in one call
- Uses `all-MiniLM-L6-v2` for embeddings

**AIGenerator** (`ai_generator.py`) - Claude API interface
- Implements Anthropic's tool use pattern
- Makes two API calls per tool-using query:
  1. Initial call where Claude decides to use search tool
  2. Follow-up call with tool results to generate final answer
- System prompt guides Claude on when to search vs use general knowledge

**ToolManager & CourseSearchTool** (`search_tools.py`)
- Implements tool interface for Claude
- `CourseSearchTool` exposes three parameters:
  - `query`: What to search for (required)
  - `course_name`: Optional filter (uses semantic matching, not exact)
  - `lesson_number`: Optional filter (exact match)
- Tracks sources from searches for UI display

**DocumentProcessor** (`document_processor.py`)
- Extracts structured course data from text files
- Uses regex to parse course metadata (title, instructor, lessons)
- Implements sentence-based chunking with overlap (800 chars, 100 overlap)
- Processes files from `docs/` folder on startup

**SessionManager** (`session_manager.py`)
- Maintains conversation history per session (max 2 exchanges by default)
- Provides context for follow-up questions

### Configuration

All settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 (text chunk size for vectors)
- `CHUNK_OVERLAP`: 100 (overlap between chunks)
- `MAX_RESULTS`: 5 (max search results)
- `MAX_HISTORY`: 2 (conversation exchanges to remember)
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"

### Data Models

See `backend/models.py`:
- **Course**: title (unique ID), course_link, instructor, lessons[]
- **Lesson**: lesson_number, title, lesson_link
- **CourseChunk**: content, course_title, lesson_number, chunk_index

### Document Loading

- Documents loaded from `../docs/` on app startup (`app.py:88-98`)
- Supports `.pdf`, `.docx`, `.txt` files
- Skips courses that already exist (checks by title)
- Use `clear_existing=True` in `add_course_folder()` to force rebuild

### Frontend

Static files in `frontend/`:
- `index.html`: Main UI
- `script.js`: API calls and response rendering
- `style.css`: Styling
- Served via FastAPI's StaticFiles at root path

## Key Implementation Notes

1. **Course Name Matching**: When filtering by course name, the system does semantic search on the `course_catalog` collection first to find the exact course title, then filters content by that exact title. This allows partial/fuzzy course name matching.

2. **Tool Use Pattern**: The AI can choose NOT to search if answering from general knowledge. Only course-specific questions trigger the search tool.

3. **ChromaDB IDs**:
   - Course catalog: Uses `course.title` as ID
   - Course content: Uses `{course_title}_{chunk_index}` as ID

4. **Session Persistence**: Sessions exist only in memory (lost on server restart). SessionManager uses UUID for session IDs.

5. **Error Handling**: VectorStore returns `SearchResults` objects with optional error messages instead of raising exceptions.
