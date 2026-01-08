import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:

**1. Course Content Search** (`search_course_content`)
- Use for: Questions about specific topics, concepts, or detailed course content
- Searches: Inside course materials to find relevant information
- Examples: "What is prompt caching?", "How do I use tool calling?", "Explain RAG in lesson 3"

**2. Course Outline** (`get_course_outline`)
- Use for: Questions about course structure, lesson lists, what's covered, curriculum overview
- Retrieves: Complete lesson list with titles, course link, instructor, and course metadata
- **IMPORTANT**: When using this tool, you MUST include the course link in your response if the tool provides it
- Examples: "What lessons are in the MCP course?", "Show me the course outline", "What topics does the Computer Use course cover?"

Tool Usage Protocol:
- **Up to 2 sequential tool calls per query** - You can make tool calls in separate rounds
- **Chain operations** - Use results from first tool call to inform the second
- **Common multi-step patterns**:
  * Get course outline → Search specific lesson content
  * Search one course → Compare with another course
  * Find lesson by description → Retrieve detailed content
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

When to Use Tools vs. General Knowledge:
- **Use tools**: Course-specific questions, lesson details, specific content queries
- **Use general knowledge**: General educational concepts, programming fundamentals, broad technical questions
- **No meta-commentary**: Don't explain which tool you used or why

Response Protocol:
- **Brief, concise and focused** - Get to the point quickly
- **Educational** - Maintain instructional value
- **Clear** - Use accessible language
- **Example-supported** - Include relevant examples when they aid understanding
- **No reasoning process** - Provide direct answers only, no search explanations or analysis

Format Guidelines:
- **Outline responses**: Present lesson lists clearly with proper numbering. ALWAYS include the course link when provided by the tool - display it prominently near the course title
- **Content responses**: Integrate search results naturally without mentioning the search
- **No meta-commentary**: Never mention "based on the search results" or "according to the outline tool"
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional multi-round tool usage.

        Supports up to 2 sequential tool calls where Claude can reason
        about previous tool results before making additional calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content with conversation history
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history with user query
        messages = [{"role": "user", "content": query}]

        # Track rounds for multi-tool calls
        current_round = 0
        max_rounds = 2

        # Main tool calling loop
        while current_round < max_rounds:
            # Prepare API parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),  # Pass a copy to avoid mutation issues
                "system": system_content
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            response = self.client.messages.create(**api_params)

            # Check if Claude used a tool
            if response.stop_reason != "tool_use":
                # No tool use - return direct response
                return self._extract_text_response(response)

            # Tool use detected - increment round counter
            current_round += 1

            # Add Claude's tool request to message history
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Execute tools and get results
            if tool_manager:
                tool_results = self._execute_tool_round(response, tool_manager)

                if not tool_results:
                    # No tools executed - shouldn't happen, but handle gracefully
                    return "Error: Unable to execute requested tools."

                # Add tool results to message history
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                # No tool manager - can't execute tools
                return "Error: Tools requested but no tool manager available."

        # Reached max rounds - make final call WITHOUT tools
        final_params = {
            **self.base_params,
            "messages": messages.copy(),  # Pass a copy to avoid mutation issues
            "system": system_content
            # No tools parameter - force final answer
        }

        final_response = self.client.messages.create(**final_params)
        return self._extract_text_response(final_response)

    def _execute_tool_round(self, response, tool_manager) -> List[Dict]:
        """
        Execute all tools from a response and return formatted results.

        Handles tool execution errors gracefully by returning error as tool result.

        Args:
            response: API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dictionaries
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Pass error to Claude as tool result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })

        return tool_results

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from API response.

        Args:
            response: API response object

        Returns:
            Text content as string
        """
        for block in response.content:
            if hasattr(block, 'type') and block.type == "text":
                return block.text

        # Fallback if no text block found
        return ""