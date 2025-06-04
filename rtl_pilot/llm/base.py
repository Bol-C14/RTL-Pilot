"""
Base LLM interface using LangChain abstractions.

This module defines the core interface for LLM providers
with tool calling capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
import logging


class LLMResponse(BaseModel):
    """Standardized LLM response format."""
    
    content: str = Field(description="The text content of the response")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made by the LLM"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens, timing, etc.)"
    )
    success: bool = Field(default=True, description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BaseLLMInterface(ABC):
    """
    Abstract base class for LLM providers with tool calling support.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Provider-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._model: Optional[BaseChatModel] = None
        self._tools: List[BaseTool] = []
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the LLM provider.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using the LLM.
        
        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools the LLM can call
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse object with the result
        """
        pass
    
    @abstractmethod
    async def stream_response(
        self,
        messages: List[BaseMessage], 
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM.
        
        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools the LLM can call
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of the response as they become available
        """
        pass
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool that the LLM can call.
        
        Args:
            tool: The tool to register
        """
        if tool not in self._tools:
            self._tools.append(tool)
            self.logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was found and removed, False otherwise
        """
        for i, tool in enumerate(self._tools):
            if tool.name == tool_name:
                self._tools.pop(i)
                self.logger.info(f"Unregistered tool: {tool_name}")
                return True
        return False
    
    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self._tools]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "provider": self.__class__.__name__,
            "config": self.config,
            "tools_count": len(self._tools)
        }
    
    async def validate_connection(self) -> bool:
        """
        Validate that the LLM connection is working.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            test_messages = [HumanMessage(content="Hello, please respond with 'OK'")]
            response = await self.generate_response(test_messages)
            return response.success and "OK" in response.content
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    def create_system_message(self, content: str) -> SystemMessage:
        """Helper to create system messages."""
        return SystemMessage(content=content)
    
    def create_human_message(self, content: str) -> HumanMessage:
        """Helper to create human messages."""
        return HumanMessage(content=content)
    
    def create_ai_message(self, content: str) -> AIMessage:
        """Helper to create AI messages."""
        return AIMessage(content=content)
