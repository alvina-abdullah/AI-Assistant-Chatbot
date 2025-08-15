import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

@cl.on_chat_start
async def start():
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

    # Initialize external client
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    # Initialize model
    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client
    )

    # Create config
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True,
    )

    # Store in session
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)

    # Create agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model
    )
    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome to the AI Assistant! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []

    # Add user message to history
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    # Retrieve session data
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")

        # Stream the agent's response
        result = Runner.run_streamed(agent, history, run_config=config)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        # Save assistant reply to history
        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", history)

        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")
