"""
LLM provider implementations using LangChain.

This module contains concrete implementations for different LLM providers.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
import asyncio

from .base import BaseLLMInterface, LLMResponse


class OpenAIProvider(BaseLLMInterface):
    """OpenAI LLM provider using LangChain."""
    
    def __init__(self, model_name: str = "gpt-4", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = kwargs.get('base_url')
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        
    async def initialize(self) -> bool:
        """Initialize the OpenAI model."""
        try:
            self._model = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30.0,
                max_retries=3
            )
            
            # Test the connection
            is_valid = await self.validate_connection()
            if is_valid:
                self.logger.info(f"OpenAI provider initialized successfully with {self.model_name}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {e}")
            return False
    
    async def generate_response(
        self, 
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        try:
            if not self._model:
                await self.initialize()
            
            # Use registered tools if none provided
            if tools is None:
                tools = self._tools
            
            # Configure the model with tools if provided
            if tools:
                model_with_tools = self._model.bind_tools(tools)
            else:
                model_with_tools = self._model
            
            # Generate response
            result = await model_with_tools.ainvoke(messages)
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(result, 'tool_calls') and result.tool_calls:
                tool_calls = [
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "id": call.get("id")
                    }
                    for call in result.tool_calls
                ]
            
            return LLMResponse(
                content=result.content,
                tool_calls=tool_calls,
                metadata={
                    "model": self.model_name,
                    "usage": getattr(result, 'usage_metadata', {}),
                    "response_metadata": getattr(result, 'response_metadata', {})
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating OpenAI response: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def stream_response(
        self,
        messages: List[BaseMessage], 
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI."""
        try:
            if not self._model:
                await self.initialize()
            
            if tools is None:
                tools = self._tools
                
            if tools:
                model_with_tools = self._model.bind_tools(tools)
            else:
                model_with_tools = self._model
            
            async for chunk in model_with_tools.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            self.logger.error(f"Error streaming OpenAI response: {e}")
            yield f"Error: {e}"


class AnthropicProvider(BaseLLMInterface):
    """Anthropic Claude LLM provider using LangChain."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        
    async def initialize(self) -> bool:
        """Initialize the Anthropic model."""
        try:
            self._model = ChatAnthropic(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30.0,
                max_retries=3
            )
            
            is_valid = await self.validate_connection()
            if is_valid:
                self.logger.info(f"Anthropic provider initialized successfully with {self.model_name}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic provider: {e}")
            return False
    
    async def generate_response(
        self, 
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic."""
        try:
            if not self._model:
                await self.initialize()
            
            if tools is None:
                tools = self._tools
            
            if tools:
                model_with_tools = self._model.bind_tools(tools)
            else:
                model_with_tools = self._model
            
            result = await model_with_tools.ainvoke(messages)
            
            tool_calls = None
            if hasattr(result, 'tool_calls') and result.tool_calls:
                tool_calls = [
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "id": call.get("id")
                    }
                    for call in result.tool_calls
                ]
            
            return LLMResponse(
                content=result.content,
                tool_calls=tool_calls,
                metadata={
                    "model": self.model_name,
                    "usage": getattr(result, 'usage_metadata', {}),
                    "response_metadata": getattr(result, 'response_metadata', {})
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating Anthropic response: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def stream_response(
        self,
        messages: List[BaseMessage], 
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic."""
        try:
            if not self._model:
                await self.initialize()
            
            if tools is None:
                tools = self._tools
                
            if tools:
                model_with_tools = self._model.bind_tools(tools)
            else:
                model_with_tools = self._model
            
            async for chunk in model_with_tools.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            self.logger.error(f"Error streaming Anthropic response: {e}")
            yield f"Error: {e}"


class LocalProvider(BaseLLMInterface):
    """Local LLM provider using Ollama through LangChain."""
    
    def __init__(self, model_name: str = "llama2", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        self.temperature = kwargs.get('temperature', 0.1)
        
    async def initialize(self) -> bool:
        """Initialize the local model."""
        try:
            self._model = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            # Note: Local models may not support tool calling
            # We'll handle this gracefully
            self.logger.info(f"Local provider initialized with {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local provider: {e}")
            return False
    
    async def generate_response(
        self, 
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using local model."""
        try:
            if not self._model:
                await self.initialize()
            
            # Convert messages to a single prompt for local models
            prompt = self._messages_to_prompt(messages)
            
            # Local models typically don't support tool calling directly
            # We'll return the response without tool calls
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._model.invoke, prompt
            )
            
            return LLMResponse(
                content=result,
                tool_calls=None,  # Local models don't support tool calling yet
                metadata={
                    "model": self.model_name,
                    "provider": "local"
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating local response: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    async def stream_response(
        self,
        messages: List[BaseMessage], 
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from local model."""
        try:
            if not self._model:
                await self.initialize()
            
            prompt = self._messages_to_prompt(messages)
            
            # Stream from local model
            for chunk in self._model.stream(prompt):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error streaming local response: {e}")
            yield f"Error: {e}"
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a single prompt string for local models."""
        prompt_parts = []
        for message in messages:
            if hasattr(message, 'type'):
                if message.type == 'system':
                    prompt_parts.append(f"System: {message.content}")
                elif message.type == 'human':
                    prompt_parts.append(f"Human: {message.content}")
                elif message.type == 'ai':
                    prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(str(message.content))
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
