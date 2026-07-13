#!/usr/bin/env python3
"""Stage 1: bare-bones LiveKit voice interview agent.

Runs as a standard LiveKit Agents Worker with automatic dispatch: it
registers with LiveKit Cloud and is handed a job for every room that gets
created (this project only ever creates one, riya-interview-room, via
server.js). We first tried a direct `room.connect()` approach mirroring
agent.js (no Worker/dispatch), but the local semantic turn-detector model
requires a JobContext-managed inference executor internally, so this is the
documented fallback from the plan -- not a preference, a hard requirement.

Wired to real STT/LLM/TTS/VAD/turn detection with a placeholder prompt. This
stage only proves the voice loop and sequential turn-taking work end to end;
the real interviewer persona and transcript logging land in later stages.

Run with:
    python interview_agent.py dev      # local dev mode, hot reload
    python interview_agent.py download-files   # pre-fetch turn-detector weights
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import anthropic, cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID") or None

# LLM_PROVIDER selects between a free local Ollama model and Anthropic Claude
# -- swapping providers is just an .env change, no code change needed.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

PLACEHOLDER_INSTRUCTIONS = (
    "You are a voice assistant helping test a real-time interview pipeline. "
    "Keep every reply to one or two short sentences so the conversation "
    "stays natural over voice, and always wait for the other person to "
    "finish speaking before you respond."
)

GREETING_INSTRUCTIONS = (
    "Greet the candidate briefly, mention this is a connection test, and "
    "ask them to say a few words back."
)


def build_llm():
    if LLM_PROVIDER == "ollama":
        return openai.LLM.with_ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    if LLM_PROVIDER == "anthropic":
        return anthropic.LLM(model=ANTHROPIC_MODEL)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (expected 'ollama' or 'anthropic')")


def build_session() -> AgentSession:
    tts_kwargs = {"voice": CARTESIA_VOICE_ID} if CARTESIA_VOICE_ID else {}

    return AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model=DEEPGRAM_MODEL),
        llm=build_llm(),
        tts=cartesia.TTS(**tts_kwargs),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
    )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    logger.info("Joined room %r as %r", ctx.room.name, ctx.local_participant_identity)

    session = build_session()
    agent = Agent(instructions=PLACEHOLDER_INSTRUCTIONS)
    await session.start(agent, room=ctx.room)

    logger.info("Session started, sending opening greeting")
    await session.generate_reply(instructions=GREETING_INSTRUCTIONS)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
