"""Tests for CourseSearchTool execute method"""
import pytest
from unittest.mock import Mock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolDefinition:
    """Test tool definition generation"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition['name'] == 'search_course_content'
        assert 'description' in definition
        assert 'input_schema' in definition
        assert definition['input_schema']['type'] == 'object'
        assert 'query' in definition['input_schema']['properties']
        assert 'course_name' in definition['input_schema']['properties']
        assert 'lesson_number' in definition['input_schema']['properties']
        assert definition['input_schema']['required'] == ['query']


class TestCourseSearchToolExecute:
    """Test CourseSearchTool execute method with various scenarios"""

    def test_execute_with_valid_results(self, mock_vector_store, sample_search_results):
        """Test execute returns formatted results when search succeeds"""
        # Setup: vector store returns valid results
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        tool = CourseSearchTool(mock_vector_store)

        # Execute search
        result = tool.execute(query="What is MCP?")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )

        # Verify result is formatted correctly
        assert isinstance(result, str)
        assert "[Introduction to MCP" in result
        assert "Lesson 1]" in result
        assert "This is the first lesson about MCP basics" in result

    def test_execute_with_course_name_filter(self, mock_vector_store, sample_search_results):
        """Test execute with course_name parameter"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        tool = CourseSearchTool(mock_vector_store)

        # Execute with course filter
        result = tool.execute(query="What is MCP?", course_name="Introduction to MCP")

        # Verify course_name was passed to search
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name="Introduction to MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self, mock_vector_store, sample_search_results):
        """Test execute with lesson_number parameter"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        tool = CourseSearchTool(mock_vector_store)

        # Execute with lesson filter
        result = tool.execute(query="What is MCP?", lesson_number=1)

        # Verify lesson_number was passed to search
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute with both course_name and lesson_number"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        tool = CourseSearchTool(mock_vector_store)

        # Execute with both filters
        result = tool.execute(
            query="What is MCP?",
            course_name="Introduction to MCP",
            lesson_number=1
        )

        # Verify both parameters were passed
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name="Introduction to MCP",
            lesson_number=1
        )

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute returns appropriate message when no results found"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)

        # Execute search that returns empty
        result = tool.execute(query="What is nonexistent topic?")

        # Verify error message
        assert "No relevant content found" in result

    def test_execute_with_empty_results_and_course_filter(self, mock_vector_store, empty_search_results):
        """Test execute with empty results includes filter info in message"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)

        # Execute with course filter
        result = tool.execute(query="What is MCP?", course_name="Introduction to MCP")

        # Verify message includes course name
        assert "No relevant content found" in result
        assert "in course 'Introduction to MCP'" in result

    def test_execute_with_empty_results_and_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test execute with empty results includes lesson info in message"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)

        # Execute with lesson filter
        result = tool.execute(query="What is MCP?", lesson_number=5)

        # Verify message includes lesson number
        assert "No relevant content found" in result
        assert "in lesson 5" in result

    def test_execute_with_error_result(self, mock_vector_store, error_search_results):
        """Test execute handles search errors correctly"""
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)

        # Execute search that returns error
        result = tool.execute(query="What is MCP?")

        # Verify error message is returned
        assert result == error_search_results.error
        assert "Search error" in result

    def test_execute_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that last_sources is populated correctly"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        tool = CourseSearchTool(mock_vector_store)

        # Execute search
        tool.execute(query="What is MCP?")

        # Verify last_sources is populated
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]['text'] == "Introduction to MCP - Lesson 1"
        assert tool.last_sources[0]['url'] == "https://example.com/lesson-1"

    def test_execute_sources_without_lesson_number(self, mock_vector_store):
        """Test source tracking when metadata has no lesson_number"""
        # Create results without lesson numbers
        results = SearchResults(
            documents=["Some course content"],
            metadata=[{
                "course_title": "Introduction to MCP",
                "chunk_index": 0
            }],
            distances=[0.1]
        )

        mock_vector_store.search.return_value = results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)

        # Execute search
        tool.execute(query="What is MCP?")

        # Verify source doesn't include lesson info
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]['text'] == "Introduction to MCP"
        assert tool.last_sources[0]['url'] is None


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert 'search_course_content' in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]['name'] == 'search_course_content'

    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test executing a registered tool"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute tool by name
        result = manager.execute_tool('search_course_content', query="What is MCP?")

        # Verify tool was executed
        assert isinstance(result, str)
        assert "Introduction to MCP" in result

        # Verify last_executed_tool is tracked
        assert manager.last_executed_tool == 'search_course_content'

    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool('nonexistent_tool', query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources from last executed tool"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute tool
        manager.execute_tool('search_course_content', query="What is MCP?")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]['text'] == "Introduction to MCP - Lesson 1"

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources after retrieval"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-1"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute tool
        manager.execute_tool('search_course_content', query="What is MCP?")

        # Reset sources
        manager.reset_sources()

        # Verify sources are cleared
        sources = manager.get_last_sources()
        assert len(sources) == 0
        assert manager.last_executed_tool is None
