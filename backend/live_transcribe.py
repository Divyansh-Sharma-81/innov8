import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time

# --- Hyperparameters for CPU Execution ---
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5  # in seconds
CHUNK_DURATION = 2    # in seconds
MODEL_SIZE = "small.en"  # "small" is a good balance of speed and accuracy for CPU
DEVICE = "cpu"        # Set to "cpu"
COMPUTE_TYPE = "int8" # "int8" is the fastest compute type for CPU

# --- Initialization ---
audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.")

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    """Continuously listen to the microphone."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=int(SAMPLE_RATE * BLOCK_DURATION), callback=audio_callback):
        print("#" * 80)
        print("Press Ctrl+C to stop the recording")
        print("#" * 80)
        while True:
            time.sleep(1)

def transcriber():
    """Continuously transcribe the audio."""
    global audio_buffer
    while True:
        try:
            # Get audio from the queue
            audio_chunk = audio_queue.get()
            audio_buffer = np.append(audio_buffer, audio_chunk)

            # Transcribe when the buffer is long enough
            if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                segments, info = model.transcribe(audio_buffer, beam_size=5)
                for segment in segments:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                # Clear the buffer after transcription
                audio_buffer = np.array([], dtype=np.float32)

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\nStopping transcriber...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    # Start the recorder in a separate thread
    recorder_thread = threading.Thread(target=recorder)
    recorder_thread.daemon = True
    recorder_thread.start()

    # Start the transcriber in the main thread
    transcriber()