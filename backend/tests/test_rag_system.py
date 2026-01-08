"""Tests for RAG system query handling"""
import pytest
from unittest.mock import Mock, patch
from rag_system import RAGSystem
from dataclasses import dataclass


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test_api_key"
    ANTHROPIC_MODEL: str = "test_model"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"
    EMBEDDING_MODEL: str = "test-embedding-model"


class TestRAGSystemInit:
    """Test RAGSystem initialization"""

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_init_creates_all_components(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                        mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test that RAGSystem initializes all required components"""
        config = MockConfig()

        system = RAGSystem(config)

        # Verify all components are initialized
        mock_doc_proc.assert_called_once_with(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        mock_vector.assert_called_once_with(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        mock_ai_gen.assert_called_once_with(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        mock_session.assert_called_once_with(config.MAX_HISTORY)
        mock_tool_mgr.assert_called()


class TestRAGSystemQuery:
    """Test RAGSystem query method"""

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session_id(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                     mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test query processing without session ID"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "AI generated response"

        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = []
        mock_tool_mgr_instance.get_last_sources.return_value = []

        # Execute query without session
        response, sources = system.query("What is MCP?")

        # Verify AI generator was called without history
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert call_args[1]['conversation_history'] is None

        # Verify response is returned
        assert response == "AI generated response"
        assert sources == []

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session_id(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                   mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test query processing with session ID"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        mock_session_instance = mock_session.return_value
        mock_session_instance.get_conversation_history.return_value = "Previous conversation"

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "AI response with context"

        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = []
        mock_tool_mgr_instance.get_last_sources.return_value = []

        # Execute query with session
        response, sources = system.query("What is MCP?", session_id="session_123")

        # Verify conversation history was retrieved
        mock_session_instance.get_conversation_history.assert_called_once_with("session_123")

        # Verify AI generator received history
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert call_args[1]['conversation_history'] == "Previous conversation"

        # Verify session was updated with exchange
        mock_session_instance.add_exchange.assert_called_once_with(
            "session_123",
            "What is MCP?",
            "AI response with context"
        )

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_passes_tools_to_ai(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                     mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test that tool definitions are passed to AI generator"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        tool_definitions = [{"name": "search_course_content"}]
        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = tool_definitions
        mock_tool_mgr_instance.get_last_sources.return_value = []

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response"

        # Execute query
        system.query("What is MCP?")

        # Verify tools were passed
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert call_args[1]['tools'] == tool_definitions
        assert call_args[1]['tool_manager'] == mock_tool_mgr_instance

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_retrieves_and_resets_sources(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                               mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test that sources are retrieved from ToolManager and then reset"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        sources = [{"text": "Source 1", "url": "http://example.com"}]
        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = []
        mock_tool_mgr_instance.get_last_sources.return_value = sources

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response"

        # Execute query
        response, returned_sources = system.query("What is MCP?")

        # Verify sources were retrieved
        mock_tool_mgr_instance.get_last_sources.assert_called_once()
        assert returned_sources == sources

        # Verify sources were reset
        mock_tool_mgr_instance.reset_sources.assert_called_once()

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_formats_prompt_correctly(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                           mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test that query is formatted into proper prompt"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = []
        mock_tool_mgr_instance.get_last_sources.return_value = []

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response"

        # Execute query
        system.query("What is MCP?")

        # Verify prompt was formatted
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert "Answer this question about course materials: What is MCP?" in call_args[1]['query']

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_error_propagation(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                    mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test that errors from AI generator propagate correctly"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = []

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.side_effect = Exception("API error")

        # Verify exception propagates
        with pytest.raises(Exception) as exc_info:
            system.query("What is MCP?")

        assert "API error" in str(exc_info.value)


class TestRAGSystemIntegration:
    """Integration tests for RAG system query flow"""

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_full_query_flow_with_tool_use(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                          mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test complete query flow when AI uses search tool"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup complex mock scenario
        mock_session_instance = mock_session.return_value
        mock_session_instance.get_conversation_history.return_value = "Previous: Hello\nAI: Hi there"

        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        mock_tool_mgr_instance.get_last_sources.return_value = [
            {"text": "MCP Course - Lesson 1", "url": "http://example.com/lesson1"}
        ]

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "MCP stands for Model Context Protocol"

        # Execute query
        response, sources = system.query("What is MCP?", session_id="session_456")

        # Verify complete flow
        # 1. History retrieved
        mock_session_instance.get_conversation_history.assert_called_once_with("session_456")

        # 2. Tools retrieved
        mock_tool_mgr_instance.get_tool_definitions.assert_called_once()

        # 3. AI generator called with all parameters
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert "What is MCP?" in call_args[1]['query']
        assert call_args[1]['conversation_history'] == "Previous: Hello\nAI: Hi there"
        assert call_args[1]['tools'] == [{"name": "search_course_content"}]
        assert call_args[1]['tool_manager'] == mock_tool_mgr_instance

        # 4. Sources retrieved and reset
        mock_tool_mgr_instance.get_last_sources.assert_called_once()
        mock_tool_mgr_instance.reset_sources.assert_called_once()

        # 5. Session updated
        mock_session_instance.add_exchange.assert_called_once_with(
            "session_456",
            "What is MCP?",
            "MCP stands for Model Context Protocol"
        )

        # 6. Correct response and sources returned
        assert response == "MCP stands for Model Context Protocol"
        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course - Lesson 1"

    @patch('rag_system.CourseOutlineTool')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session_no_history_update(self, mock_doc_proc, mock_vector, mock_ai_gen,
                                                     mock_session, mock_tool_mgr, mock_search_tool, mock_outline_tool):
        """Test that session is not updated when no session_id provided"""
        config = MockConfig()
        system = RAGSystem(config)

        # Setup mocks
        mock_session_instance = mock_session.return_value
        mock_tool_mgr_instance = mock_tool_mgr.return_value
        mock_tool_mgr_instance.get_tool_definitions.return_value = []
        mock_tool_mgr_instance.get_last_sources.return_value = []

        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response"

        # Execute query without session_id
        system.query("What is MCP?")

        # Verify session was not updated
        mock_session_instance.add_exchange.assert_not_called()
        mock_session_instance.get_conversation_history.assert_not_called()
