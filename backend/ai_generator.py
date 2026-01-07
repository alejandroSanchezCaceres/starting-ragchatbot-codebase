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
- **One tool call per query maximum** - Choose the most appropriate tool
- **Search for content**, **outline for structure** - Don't mix use cases
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
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text