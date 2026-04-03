#!/usr/bin/env python3
"""Test client for Qwen2.5-Omni API."""

import requests
import json
import sys

def test_text_query(prompt: str = "Hello! How are you today?"):
    """Test basic text query."""
    
    print(f"🔹 Testing text query: '{prompt}'")
    
    payload = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        message = result["choices"][0]["message"]["content"]
        
        print(f"✅ Response: {message}\n")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}\n")
        return None


def test_audio_query(audio_url: str, prompt: str = "What is in this audio?"):
    """Test audio + text query."""
    
    print(f"🔹 Testing audio query: '{prompt}'")
    print(f"   Audio URL: {audio_url}")
    
    payload = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        message = result["choices"][0]["message"]["content"]
        
        print(f"✅ Response: {message}\n")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}\n")
        return None


def test_streaming(prompt: str = "Count from 1 to 10 slowly."):
    """Test streaming response."""
    
    print(f"🔹 Testing streaming: '{prompt}'")
    
    payload = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True,
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        print("✅ Streaming response: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print("\n")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}\n")
        return None


def check_server_health():
    """Check if server is running."""
    
    print("🔹 Checking server health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy\n")
            return True
        else:
            print(f"⚠️  Server returned status {response.status_code}\n")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not reachable: {e}\n")
        return False


def main():
    """Run all tests."""
    
    print("=" * 60)
    print("Qwen2.5-Omni API Test Client")
    print("=" * 60 + "\n")
    
    # Check server health
    if not check_server_health():
        print("❌ Server is not running. Start it with: uv run qwen3-serve")
        sys.exit(1)
    
    # Test 1: Simple text query
    test_text_query("What is the capital of France?")
    
    # Test 2: Streaming
    test_streaming("Tell me a short joke.")
    
    # Test 3: Audio (example - replace with actual audio URL)
    # test_audio_query(
    #     "https://example.com/audio.wav",
    #     "Transcribe this audio."
    # )
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
