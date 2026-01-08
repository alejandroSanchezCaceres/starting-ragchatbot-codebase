"""Pytest configuration and shared fixtures for RAG system tests"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys
import os

# Add backend directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp-course",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/lesson-1"
            ),
            Lesson(
                lesson_number=2,
                title="Advanced Topics",
                lesson_link="https://example.com/lesson-2"
            )
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Introduction to MCP Lesson 1 content: This is the first lesson about MCP basics.",
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Introduction to MCP Lesson 1 content: This covers advanced MCP concepts.",
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Introduction to MCP Lesson 2 content: Deep dive into MCP architecture.",
            course_title="Introduction to MCP",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_search_results():
    """Create sample search results"""
    return SearchResults(
        documents=[
            "This is the first lesson about MCP basics.",
            "This covers advanced MCP concepts."
        ],
        metadata=[
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Introduction to MCP",
                "lesson_number": 1,
                "chunk_index": 1
            }
        ],
        distances=[0.15, 0.23]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Create search results with error"""
    return SearchResults.empty("Search error: ChromaDB connection failed")


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore"""
    mock = Mock()
    mock.search = Mock()
    mock.get_lesson_link = Mock(return_value="https://example.com/lesson-1")
    mock.max_results = 5
    return mock


@pytest.fixture
def mock_chroma_collection():
    """Create a mock ChromaDB collection"""
    mock = Mock()
    mock.query = Mock()
    mock.add = Mock()
    mock.get = Mock()
    return mock


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client"""
    mock = Mock()
    mock.get_or_create_collection = Mock()
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock = Mock()
    mock.messages = Mock()
    mock.messages.create = Mock()
    return mock


@pytest.fixture
def mock_anthropic_response_no_tool():
    """Create a mock Anthropic response without tool use"""
    mock = Mock()
    mock.stop_reason = "end_turn"
    mock.content = [Mock(type="text", text="This is the response text")]
    return mock


@pytest.fixture
def mock_anthropic_response_with_tool():
    """Create a mock Anthropic response with tool use"""
    mock = Mock()
    mock.stop_reason = "tool_use"

    # Create mock tool use block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "What is MCP?"}

    mock.content = [tool_block]
    return mock


@pytest.fixture
def mock_anthropic_final_response():
    """Create a mock final Anthropic response after tool use"""
    mock = Mock()
    mock.stop_reason = "end_turn"
    mock.content = [Mock(type="text", text="Based on the search results, MCP is...")]
    return mock


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    mock = Mock()
    mock.get_tool_definitions = Mock(return_value=[
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ])
    mock.execute_tool = Mock(return_value="Search results: MCP is Model Context Protocol")
    mock.get_last_sources = Mock(return_value=[])
    mock.reset_sources = Mock()
    return mock


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager"""
    mock = Mock()
    mock.create_session = Mock(return_value="session_123")
    mock.get_conversation_history = Mock(return_value="Previous conversation context")
    mock.add_exchange = Mock()
    mock.clear_session = Mock()
    return mock


@pytest.fixture
def mock_ai_generator():
    """Create a mock AIGenerator"""
    mock = Mock()
    mock.generate_response = Mock(return_value="Generated response from AI")
    return mock


@pytest.fixture
def create_tool_use_response():
    """Factory fixture to create mock tool use responses"""
    def _create(tool_name: str, tool_input: dict, tool_id: str = None):
        mock = Mock()
        mock.stop_reason = "tool_use"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = tool_id or f"tool_{tool_name}"
        tool_block.name = tool_name
        tool_block.input = tool_input

        mock.content = [tool_block]
        return mock

    return _create


@pytest.fixture
def create_text_response():
    """Factory fixture to create mock text responses"""
    def _create(text: str):
        mock = Mock()
        mock.stop_reason = "end_turn"

        text_block = Mock()
        text_block.type = "text"
        text_block.text = text

        mock.content = [text_block]
        return mock

    return _create
