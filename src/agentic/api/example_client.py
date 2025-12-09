#!/usr/bin/env python3
"""
Example client for the OriginHub Agentic System API.
Shows how to connect to the API from your UI.
"""

import requests
import json
from typing import Optional

class OriginHubClient:
    """Client for interacting with OriginHub API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
    
    def health_check(self) -> dict:
        """Check if API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_session(self) -> str:
        """Create a new conversation session."""
        response = requests.post(f"{self.base_url}/sessions")
        response.raise_for_status()
        data = response.json()
        self.session_id = data["session_id"]
        print(f"✓ Session created: {self.session_id}")
        return self.session_id
    
    def send_message(self, message: str) -> dict:
        """Send a message to the API."""
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")
        
        response = requests.post(
            f"{self.base_url}/chat/{self.session_id}",
            json={"message": message}
        )
        response.raise_for_status()
        return response.json()
    
    def get_session_info(self) -> dict:
        """Get current session information."""
        if not self.session_id:
            raise ValueError("No active session.")
        
        response = requests.get(f"{self.base_url}/sessions/{self.session_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_session(self) -> bool:
        """Delete the current session."""
        if not self.session_id:
            return False
        
        response = requests.delete(f"{self.base_url}/sessions/{self.session_id}")
        response.raise_for_status()
        self.session_id = None
        return True
    
    def interactive_chat(self):
        """Interactive chat loop."""
        print("\n" + "="*60)
        print("OriginHub AI Client")
        print("="*60)
        print("Type 'exit', 'quit', or Ctrl+C to end the conversation.\n")
        
        self.create_session()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    self.delete_session()
                    break
                
                if not user_input:
                    continue
                
                print("\nProcessing...")
                result = self.send_message(user_input)
                
                # Pretty print response
                response = result.get("response")
                if isinstance(response, dict):
                    print(f"\nAssistant: {json.dumps(response, indent=2)}\n")
                else:
                    print(f"\nAssistant: {response}\n")
                
                # Show if analysis is complete
                if result.get("analysis_complete"):
                    print("(Analysis complete - you can ask follow-up questions)\n")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                self.delete_session()
                break
            except requests.exceptions.RequestException as e:
                print(f"\n✗ API Error: {str(e)}\n")
            except Exception as e:
                print(f"\n✗ Error: {str(e)}\n")


def main():
    """Run the client."""
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Connecting to API at {api_url}...")
    
    client = OriginHubClient(api_url)
    
    # Check health first
    try:
        health = client.health_check()
        if health["status"] != "healthy":
            print(f"✗ API not healthy: {health}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to API: {str(e)}")
        sys.exit(1)
    
    print("✓ Connected to API")
    
    # Start interactive chat
    client.interactive_chat()


if __name__ == "__main__":
    main()
