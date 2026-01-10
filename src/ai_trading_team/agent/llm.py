"""LLM client configuration for langchain-anthropic."""

from ai_trading_team.config import Config


def create_llm(config: Config):  # type: ignore[no-untyped-def]
    """Create LangChain LLM client.

    Args:
        config: Application configuration

    Returns:
        Configured ChatAnthropic instance

    Note:
        Uses third-party API endpoint from config (e.g., api.minimax.io)
    """
    # Lazy import to avoid loading langchain when not needed
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=config.api.anthropic_api_key,
        anthropic_api_url=config.api.anthropic_base_url,
        max_tokens=4096,
    )
