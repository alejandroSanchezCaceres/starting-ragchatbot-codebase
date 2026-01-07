from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI with URL
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"

            # Get lesson link from vector store
            source_url = None
            if lesson_num is not None:
                source_url = self.store.get_lesson_link(course_title, lesson_num)

            sources.append({
                "text": source_text,
                "url": source_url
            })

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving complete course outline with lesson structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track course link for UI

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for course outline retrieval"""
        return {
            "name": "get_course_outline",
            "description": "Retrieve the complete structure and lesson list for a specific course. Use this when users ask about course organization, lesson titles, what's covered, or the full curriculum of a course.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title or partial course name (e.g., 'MCP', 'Computer Use'). Partial matches work via semantic search."
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Retrieve and format complete course outline.

        Args:
            course_name: Full or partial course title

        Returns:
            Formatted course outline with title, instructor, and all lessons
        """
        # Step 1: Resolve course name using semantic search
        resolved_title = self.store._resolve_course_name(course_name)

        if not resolved_title:
            return f"No course found matching '{course_name}'. Please try a different course name."

        # Step 2: Get course metadata from catalog
        try:
            results = self.store.course_catalog.get(ids=[resolved_title])

            if not results or not results['metadatas'] or not results['metadatas'][0]:
                return f"Course '{resolved_title}' found but no metadata available."

            metadata = results['metadatas'][0]

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"

        # Step 3: Format and return outline
        return self._format_outline(metadata)

    def _format_outline(self, metadata: Dict[str, Any]) -> str:
        """
        Format course metadata into structured outline.

        Args:
            metadata: Course metadata dict with title, instructor, course_link, lessons_json

        Returns:
            Formatted text outline
        """
        import json

        # Extract basic info
        title = metadata.get('title', 'Unknown Course')
        instructor = metadata.get('instructor', 'Unknown Instructor')
        course_link = metadata.get('course_link')
        lessons_json = metadata.get('lessons_json', '[]')

        # Parse lessons
        try:
            lessons = json.loads(lessons_json)
        except json.JSONDecodeError:
            lessons = []

        # Track source for UI (course link)
        if course_link:
            self.last_sources = [{
                "text": title,
                "url": course_link
            }]
        else:
            self.last_sources = []

        # Build formatted outline
        outline_parts = []

        # Add course title with link if available
        if course_link:
            outline_parts.append(f"**Course:** [{title}]({course_link})")
        else:
            outline_parts.append(f"**Course:** {title}")

        outline_parts.extend([
            f"**Instructor:** {instructor}",
            "",
            "**Lessons:**"
        ])

        if lessons:
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number', '?')
                lesson_title = lesson.get('lesson_title', 'Unknown')
                outline_parts.append(f"  {lesson_num}. {lesson_title}")
        else:
            outline_parts.append("  No lessons available")

        return "\n".join(outline_parts)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}
        self.last_executed_tool = None  # Track which tool was executed last
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        # Track which tool is being executed
        self.last_executed_tool = tool_name
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last executed tool"""
        # If we know which tool was executed last, get its sources
        if self.last_executed_tool and self.last_executed_tool in self.tools:
            tool = self.tools[self.last_executed_tool]
            if hasattr(tool, 'last_sources'):
                return tool.last_sources

        # Fallback: check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
        # Also reset the last executed tool tracker
        self.last_executed_tool = None