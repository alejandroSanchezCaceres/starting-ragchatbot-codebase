"""Tests for AIGenerator tool calling behavior"""
import pytest
from unittest.mock import Mock, patch, call
from ai_generator import AIGenerator


class TestAIGeneratorBasic:
    """Test basic AIGenerator functionality"""

    def test_init(self):
        """Test AIGenerator initialization"""
        api_key = "test_api_key"
        model = "claude-sonnet-4-20250514"

        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key, model)

            assert generator.model == model
            assert generator.base_params['model'] == model
            assert generator.base_params['temperature'] == 0
            assert generator.base_params['max_tokens'] == 800
            mock_anthropic.assert_called_once_with(api_key=api_key)


class TestAIGeneratorWithoutTools:
    """Test AIGenerator response generation without tool use"""

    def test_generate_response_without_tools(self, mock_anthropic_client, mock_anthropic_response_no_tool):
        """Test generating response when no tools are provided"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_response_no_tool

            result = generator.generate_response(query="What is MCP?")

            # Verify API was called without tools
            call_args = mock_anthropic_client.messages.create.call_args
            assert 'tools' not in call_args[1]
            assert call_args[1]['messages'][0]['content'] == "What is MCP?"

            # Verify response text is returned
            assert result == "This is the response text"

    def test_generate_response_includes_conversation_history(self, mock_anthropic_client, mock_anthropic_response_no_tool):
        """Test that conversation history is included in system prompt"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_response_no_tool

            history = "User: Previous question\nAssistant: Previous answer"
            result = generator.generate_response(
                query="What is MCP?",
                conversation_history=history
            )

            # Verify system prompt includes history
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]['system']
            assert "Previous conversation:" in system_content
            assert history in system_content


class TestAIGeneratorWithTools:
    """Test AIGenerator tool calling behavior"""

    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client, mock_anthropic_response_no_tool, mock_tool_manager):
        """Test response when tools provided but Claude doesn't use them"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_response_no_tool

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is 2+2?",
                tools=tools
            )

            # Verify tools were passed to API
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args[1]['tools'] == tools
            assert call_args[1]['tool_choice'] == {"type": "auto"}

            # Since no tool use, should return direct response
            assert result == "This is the response text"

    def test_generate_response_with_tool_use(self, mock_anthropic_client, mock_anthropic_response_with_tool,
                                            mock_anthropic_final_response, mock_tool_manager):
        """Test response when Claude uses a tool"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # First call returns tool use, second call returns final response
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify two API calls were made
            assert mock_anthropic_client.messages.create.call_count == 2

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                'search_course_content',
                query="What is MCP?"
            )

            # Verify final response is returned
            assert result == "Based on the search results, MCP is..."

    def test_tool_execution_messages_format(self, mock_anthropic_client, mock_anthropic_response_with_tool,
                                           mock_anthropic_final_response, mock_tool_manager):
        """Test that messages are correctly formatted during tool execution"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Setup responses
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check second API call (final response)
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            messages = second_call[1]['messages']

            # Should have 3 messages: user query, assistant tool use, user tool results
            assert len(messages) == 3

            # First message: user query
            assert messages[0]['role'] == 'user'
            assert messages[0]['content'] == "What is MCP?"

            # Second message: assistant tool use
            assert messages[1]['role'] == 'assistant'
            assert messages[1]['content'] == mock_anthropic_response_with_tool.content

            # Third message: user with tool results
            assert messages[2]['role'] == 'user'
            assert isinstance(messages[2]['content'], list)
            assert messages[2]['content'][0]['type'] == 'tool_result'
            assert messages[2]['content'][0]['tool_use_id'] == 'tool_123'

    def test_tool_result_content(self, mock_anthropic_client, mock_anthropic_response_with_tool,
                                mock_anthropic_final_response, mock_tool_manager):
        """Test that tool results are correctly included in message"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Setup tool manager to return specific result
            mock_tool_manager.execute_tool.return_value = "Search results: MCP is Model Context Protocol"

            # Setup responses
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check tool result content in second API call
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            tool_results = second_call[1]['messages'][2]['content']

            assert tool_results[0]['content'] == "Search results: MCP is Model Context Protocol"

    def test_multiple_tool_uses(self, mock_anthropic_client, mock_anthropic_final_response, mock_tool_manager):
        """Test handling multiple tool calls in one response"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Create response with multiple tool uses
            multi_tool_response = Mock()
            multi_tool_response.stop_reason = "tool_use"

            tool_block_1 = Mock()
            tool_block_1.type = "tool_use"
            tool_block_1.id = "tool_123"
            tool_block_1.name = "search_course_content"
            tool_block_1.input = {"query": "What is MCP?"}

            tool_block_2 = Mock()
            tool_block_2.type = "tool_use"
            tool_block_2.id = "tool_456"
            tool_block_2.name = "search_course_content"
            tool_block_2.input = {"query": "What is Claude?"}

            multi_tool_response.content = [tool_block_1, tool_block_2]

            mock_anthropic_client.messages.create.side_effect = [
                multi_tool_response,
                mock_anthropic_final_response
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP and Claude?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2

            # Verify both tool results are in second API call
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            tool_results = second_call[1]['messages'][2]['content']
            assert len(tool_results) == 2

    def test_no_tools_parameter_in_final_call(self, mock_anthropic_client, mock_anthropic_response_with_tool,
                                             mock_anthropic_final_response, mock_tool_manager):
        """Test that tools are still available in round 2 with sequential calling"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # With sequential tool calling, tools are still available in second call
            # (Claude just chose not to use them - early termination)
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            assert 'tools' in second_call[1]  # Tools still available
            assert 'tool_choice' in second_call[1]


class TestAIGeneratorSystemPrompt:
    """Test system prompt behavior"""

    def test_system_prompt_content(self, mock_anthropic_client, mock_anthropic_response_no_tool):
        """Test that system prompt contains expected instructions"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_response_no_tool

            result = generator.generate_response(query="What is MCP?")

            # Verify system prompt includes key instructions
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]['system']

            assert "course materials" in system_content.lower()
            assert "search_course_content" in system_content
            assert "get_course_outline" in system_content

    def test_system_prompt_without_history(self, mock_anthropic_client, mock_anthropic_response_no_tool):
        """Test system prompt when no conversation history"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_response_no_tool

            result = generator.generate_response(query="What is MCP?")

            # Verify system prompt doesn't include "Previous conversation" section
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]['system']

            assert system_content == AIGenerator.SYSTEM_PROMPT


class TestAIGeneratorSequentialToolCalls:
    """Test sequential tool calling behavior"""

    def test_two_sequential_tool_calls(self, mock_anthropic_client, mock_tool_manager,
                                       create_tool_use_response, create_text_response):
        """Test Claude making 2 tool calls in sequence"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Setup: 3 API calls
            # 1. Initial call → tool_use (get_course_outline)
            # 2. After tool result → tool_use (search_course_content)
            # 3. After second tool result → end_turn (final answer)
            mock_anthropic_client.messages.create.side_effect = [
                create_tool_use_response("get_course_outline", {"course_name": "MCP"}),
                create_tool_use_response("search_course_content", {"query": "lesson 4"}),
                create_text_response("Based on the outline and search, lesson 4 covers...")
            ]

            # Setup tool manager to return different results
            mock_tool_manager.execute_tool.side_effect = [
                "Lesson 4: Advanced MCP Features",
                "Search results: MCP advanced features include..."
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What does lesson 4 of MCP cover?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify 3 API calls made
            assert mock_anthropic_client.messages.create.call_count == 3

            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 4")

            # Verify final response
            assert "lesson 4 covers" in result.lower()

    def test_early_termination_after_first_tool(self, mock_anthropic_client, mock_tool_manager,
                                                create_tool_use_response, create_text_response):
        """Test termination when Claude doesn't use tool in round 2"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Setup: 2 API calls (Claude gets enough info from first tool)
            # 1. Initial call → tool_use
            # 2. After tool result → end_turn (Claude has enough info)
            mock_anthropic_client.messages.create.side_effect = [
                create_tool_use_response("search_course_content", {"query": "MCP"}),
                create_text_response("MCP stands for Model Context Protocol...")
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify only 2 API calls made (not 3)
            assert mock_anthropic_client.messages.create.call_count == 2

            # Verify only one tool executed
            assert mock_tool_manager.execute_tool.call_count == 1

            # Verify response
            assert "MCP stands for" in result

    def test_max_rounds_enforced_with_final_call(self, mock_anthropic_client, mock_tool_manager,
                                                 create_tool_use_response, create_text_response):
        """Test that max 2 rounds is enforced with final API call"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Setup: Claude uses tools in both rounds, forced to answer in third
            mock_anthropic_client.messages.create.side_effect = [
                create_tool_use_response("get_course_outline", {"course_name": "MCP"}),
                create_tool_use_response("search_course_content", {"query": "lesson 4"}),
                create_text_response("Final answer based on two tool calls")
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="Complex query requiring multiple tools",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify 3 API calls made
            assert mock_anthropic_client.messages.create.call_count == 3

            # Verify tools present in first 2 calls, absent in final call
            calls = mock_anthropic_client.messages.create.call_args_list
            assert 'tools' in calls[0][1]  # First call has tools
            assert 'tools' in calls[1][1]  # Second call has tools
            assert 'tools' not in calls[2][1]  # Final call NO tools (forced answer)

            # Verify both tools executed
            assert mock_tool_manager.execute_tool.call_count == 2

            # Verify result
            assert result == "Final answer based on two tool calls"

    def test_message_accumulation_across_rounds(self, mock_anthropic_client, mock_tool_manager,
                                                create_tool_use_response, create_text_response):
        """Test that messages accumulate correctly across rounds"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            mock_anthropic_client.messages.create.side_effect = [
                create_tool_use_response("get_course_outline", {"course_name": "MCP"}),
                create_tool_use_response("search_course_content", {"query": "lesson 4"}),
                create_text_response("Final answer")
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="Test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check message accumulation in each API call
            calls = mock_anthropic_client.messages.create.call_args_list

            # Call 1: Initial query only
            assert len(calls[0][1]['messages']) == 1
            assert calls[0][1]['messages'][0]['role'] == 'user'

            # Call 2: query + assistant_tool_use_1 + user_tool_result_1
            assert len(calls[1][1]['messages']) == 3
            assert calls[1][1]['messages'][0]['role'] == 'user'
            assert calls[1][1]['messages'][1]['role'] == 'assistant'
            assert calls[1][1]['messages'][2]['role'] == 'user'

            # Call 3: all previous + assistant_tool_use_2 + user_tool_result_2
            assert len(calls[2][1]['messages']) == 5
            assert calls[2][1]['messages'][0]['role'] == 'user'
            assert calls[2][1]['messages'][1]['role'] == 'assistant'
            assert calls[2][1]['messages'][2]['role'] == 'user'
            assert calls[2][1]['messages'][3]['role'] == 'assistant'
            assert calls[2][1]['messages'][4]['role'] == 'user'

    def test_tool_execution_error_passed_to_claude(self, mock_anthropic_client, mock_tool_manager,
                                                   create_tool_use_response, create_text_response):
        """Test that tool errors are passed to Claude as tool results"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_key", "test_model")

            # Tool execution fails
            mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

            mock_anthropic_client.messages.create.side_effect = [
                create_tool_use_response("search_course_content", {"query": "MCP"}),
                create_text_response("I encountered an error searching the database...")
            ]

            tools = mock_tool_manager.get_tool_definitions()
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify error didn't crash the system
            assert "error" in result.lower()

            # Verify tool error was passed to Claude as a tool result
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            tool_result_message = second_call[1]['messages'][2]

            assert tool_result_message['role'] == 'user'
            assert tool_result_message['content'][0]['type'] == 'tool_result'
            assert 'Error executing tool' in tool_result_message['content'][0]['content']
            assert 'Database connection failed' in tool_result_message['content'][0]['content']
