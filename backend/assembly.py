#!/usr/bin/env python3
import os, logging
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions,
    StreamingEvents, StreamingParameters,
    BeginEvent, TurnEvent, TerminationEvent, StreamingError
)
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set ASSEMBLYAI_API_KEY in env before running")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aai-stream")

def on_begin(self, event: BeginEvent):
    logger.info(f"Session started: {event.id}")

def on_turn(self, event: TurnEvent):
    # event.transcript is immutable finalized text for the turn
    print("\n--- TURN ---")
    print(event.transcript)
    # you can inspect event.words for timestamps/confidence
    # if you want formatted text only when available:
    if event.turn_is_formatted:
        print("[formatted]")

def on_termination(self, event: TerminationEvent):
    logger.info(f"Session terminated — audio processed: {event.audio_duration_seconds}s")

def on_error(self, error: StreamingError):
    logger.error(f"Streaming error: {error}")

def main():
    aai.settings.api_key = API_KEY  # optional; StreamingClient also accepts api_key param
    client = StreamingClient(
        StreamingClientOptions(api_key=API_KEY, api_host="streaming.assemblyai.com")
    )

    # bind handlers
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_termination)
    client.on(StreamingEvents.Error, on_error)

    # connect with parameters (mono 16k recommended)
    client.connect(StreamingParameters(sample_rate=16000, format_turns=True))

    try:
        # MicrophoneStream handles the mic capture and yields properly chunked audio
        client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
