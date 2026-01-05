import json
from typing import Callable, get_args, get_origin, Union
import inspect
from typing import _GenericAlias
import asyncio
from functools import wraps


def get_fn_signature(fn: Callable) -> dict:
    """
    Generates the signature for a given function.

    Args:
        fn (Callable): The function whose signature needs to be extracted.

    Returns:
        dict: A dictionary containing the function's name, description,
              and parameter types with optional status and default values.
    """
    fn_signature: dict = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {"properties": {}},
        "is_async": asyncio.iscoroutinefunction(fn)  # Add flag for async functions
    }
    
    sig = inspect.signature(fn)
    schema = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'return':
            continue
            
        annotation = fn.__annotations__.get(param_name)
        if annotation is None:
            continue
            
        param_info = {}
        
        # Handle Optional types and extract the actual type
        if isinstance(annotation, _GenericAlias) and get_origin(annotation) is Union:
            types = get_args(annotation)
            # Filter out NoneType to get the actual type
            base_type = next(t for t in types if t is not type(None))
            param_info["type"] = base_type.__name__
            param_info["optional"] = True
        else:
            param_info["type"] = annotation.__name__
            param_info["optional"] = False
        
        # Add default value if present
        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default
            
        schema[param_name] = param_info
    
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    Validates and converts arguments in the input dictionary to match the expected types.
    Handles optional arguments and default values.

    Args:
        tool_call (dict): A dictionary containing the arguments passed to the tool.
        tool_signature (dict): The expected function signature and parameter types.

    Returns:
        dict: The tool call dictionary with the arguments converted to the correct types if necessary.
    """
    properties = tool_signature["parameters"]["properties"]
    
    if "arguments" not in tool_call:
        tool_call["arguments"] = {}

    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
    }
    
    for param_name, param_info in properties.items():
        expected_type = param_info["type"]
        
        # If parameter is not in arguments but has a default value, use it
        if param_name not in tool_call["arguments"] and "default" in param_info:
            tool_call["arguments"][param_name] = param_info["default"]
            continue
            
        # Skip if parameter is optional and not provided
        if param_info.get("optional", False) and param_name not in tool_call["arguments"]:
            continue
            
        # If parameter is provided, validate and convert type
        if param_name in tool_call["arguments"]:
            arg_value = tool_call["arguments"][param_name]
            
            # Skip None values for optional parameters
            if arg_value is None and param_info.get("optional", False):
                continue
                
            # Convert type if necessary and value is not None
            if arg_value is not None and not isinstance(arg_value, type_mapping[expected_type]):
                try:
                    tool_call["arguments"][param_name] = type_mapping[expected_type](arg_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert argument '{param_name}' to type {expected_type}: {str(e)}")

    return tool_call


class Tool:
    """
    A class representing a tool that wraps a callable and its signature.
    Supports both synchronous and asynchronous functions.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents.
        fn_signature (str): JSON string representation of the function's signature.
        is_async (bool): Whether the function is asynchronous.
    """

    def __init__(self, name: str, fn: Callable, fn_signature: str, is_async: bool):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature
        self.is_async = is_async

    def __str__(self):
        return self.fn_signature

    async def _run_async(self, **kwargs):
        """
        Internal method to run async functions.
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return await self.fn(**filtered_kwargs)

    def _run_sync(self, **kwargs):
        """
        Internal method to run sync functions.
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.fn(**filtered_kwargs)

    def run(self, **kwargs):
        """
        Executes the tool (function) with provided arguments.
        Handles both synchronous and asynchronous functions.

        Args:
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call, or a coroutine if the function is async.
        """
        if self.is_async:
            # For async functions, return a coroutine that can be awaited
            return self._run_async(**kwargs)
        else:
            # For normal functions, return the result directly
            return self._run_sync(**kwargs)


def tool(fn: Callable):
    """
    A decorator that wraps a function into a Tool object.
    Supports both synchronous and asynchronous functions.

    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Tool: A Tool object containing the function, its name, and its signature.
    """
    @wraps(fn)
    def wrapper():
        fn_signature = get_fn_signature(fn)
        is_async = asyncio.iscoroutinefunction(fn)
        return Tool(
            name=fn_signature.get("name"),
            fn=fn,
            fn_signature=json.dumps(fn_signature),
            is_async=is_async
        )

    return wrapper()