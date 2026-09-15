"""Framework-specific exceptions."""


class GeoAgentError(Exception):
    """Base error for the geo agent system."""


class ConfigError(GeoAgentError):
    """Raised when configuration cannot be loaded or validated."""


class RegistryError(GeoAgentError):
    """Raised when component registration or lookup fails."""


class ToolExecutionError(GeoAgentError):
    """Raised when a tool fails unexpectedly."""


class ImageInputError(GeoAgentError, ValueError):
    """Raised when a workflow input is not a readable, decodable image."""
