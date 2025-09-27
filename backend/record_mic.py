#!/usr/bin/env python3
import argparse, wave, pyaudio, audioop, sys, time

def record(device_index: int | None, seconds: int, out_path: str, rate: int = 16000,
           channels: int = 1, frames_per_buffer: int = 4096):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,  # 16-bit PCM
            channels=channels,       # mono
            rate=rate,               # 16 kHz (recommended)
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
        )
    except Exception as e:
        print(f"Failed to open input device {device_index}: {e}")
        pa.terminate()
        sys.exit(1)

    frames = []
    blocks = int(rate / frames_per_buffer * seconds)
    print(f"Recording {seconds}s @ {rate} Hz from device={device_index} → {out_path}")
    try:
        for i in range(blocks):
            # tolerate brief overruns instead of crashing on macOS
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            if i % 20 == 0:  # occasional RMS to confirm we’re not recording silence
                try:
                    print(f"RMS≈{audioop.rms(data, 2)}")
                except Exception:
                    pass
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    print("Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None, help="input device index (or omit to use default)")
    ap.add_argument("--seconds", type=int, default=5, help="recording length in seconds")
    ap.add_argument("--out", type=str, default="mic_test.wav", help="output WAV filename")
    args = ap.parse_args()
    record(args.device, args.seconds, args.out)
