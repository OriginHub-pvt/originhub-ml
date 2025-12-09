#!/usr/bin/env python3
"""
Complete working example of using the OriginHub API.
Shows all steps from session creation to follow-up questions.
"""

import requests
import json
from typing import Dict, Any
import time

class OriginHubAPIExample:
    """Example demonstrating full API usage."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    def check_health(self):
        """Check if API is healthy."""
        print("\n1. Checking API Health...")
        try:
            response = requests.get(f"{self.base_url}/health")
            health = response.json()
            print(f"   Status: {health['status']}")
            print(f"   Models Loaded: {health['models_loaded']}")
            print(f"   Weaviate Connected: {health['weaviate_connected']}")
            return health['status'] == 'healthy'
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def create_session(self):
        """Create a new conversation session."""
        print("\n2. Creating Session...")
        try:
            response = requests.post(f"{self.base_url}/sessions")
            data = response.json()
            self.session_id = data['session_id']
            print(f"   Session ID: {self.session_id}")
            print(f"   Created: {data['created_at']}")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def send_initial_message(self, message: str) -> Dict[str, Any]:
        """Send first message for analysis."""
        print(f"\n3. Sending Initial Message...")
        print(f"   User: {message}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/{self.session_id}",
                json={"message": message}
            )
            result = response.json()
            
            # Pretty print response
            if isinstance(result['response'], dict):
                print(f"   Assistant: {json.dumps(result['response'], indent=4)}")
            else:
                print(f"   Assistant: {result['response']}")
            
            print(f"   Analysis Complete: {result['analysis_complete']}")
            print(f"   Conversation Length: {len(result['conversation_history'])}")
            
            return result
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return None
    
    def send_followup_message(self, message: str) -> Dict[str, Any]:
        """Send follow-up question."""
        print(f"\n4. Sending Follow-up Question...")
        print(f"   User: {message}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/{self.session_id}",
                json={"message": message}
            )
            result = response.json()
            
            # Pretty print response
            if isinstance(result['response'], dict):
                print(f"   Assistant: {json.dumps(result['response'], indent=4)}")
            else:
                print(f"   Assistant: {result['response']}")
            
            return result
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return None
    
    def get_session_info(self):
        """Get session metadata."""
        print("\n5. Getting Session Info...")
        try:
            response = requests.get(f"{self.base_url}/sessions/{self.session_id}")
            info = response.json()
            print(f"   Session ID: {info['session_id']}")
            print(f"   Created: {info['created_at']}")
            print(f"   Last Activity: {info['last_activity']}")
            print(f"   Message Count: {info['message_count']}")
            return info
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return None
    
    def delete_session(self):
        """End session and cleanup."""
        print(f"\n6. Deleting Session...")
        try:
            response = requests.delete(f"{self.base_url}/sessions/{self.session_id}")
            print(f"   Session deleted successfully")
            self.session_id = None
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def run_example(self):
        """Run complete example workflow."""
        print("="*60)
        print("OriginHub API Complete Example")
        print("="*60)
        
        # Check health
        if not self.check_health():
            print("\n✗ API not healthy. Make sure it's running:")
            print("  python src/agentic/scripts/api_server.py")
            return
        
        # Create session
        if not self.create_session():
            return
        
        # Send initial business idea
        idea = """
        An AI-powered personal assistant that analyzes your sleep patterns 
        using wearable devices and wearable biometric sensors to provide 
        personalized recommendations for improving sleep quality and overall wellness.
        """
        
        result1 = self.send_initial_message(idea.strip())
        if not result1:
            return
        
        # If analysis is not complete, ask clarifier questions
        if not result1.get('analysis_complete'):
            print("\n   (Waiting for clarifications...)")
            time.sleep(2)
        
        # Send follow-up questions
        followup_questions = [
            "What's the primary revenue model for this business?",
            "How would you differentiate from existing sleep tracking apps?",
            "What's your go-to-market strategy?"
        ]
        
        for i, question in enumerate(followup_questions[:2], 1):
            print(f"\nFollowup {i}:")
            result = self.send_followup_message(question)
            if not result:
                break
            time.sleep(1)  # Rate limiting
        
        # Show session info
        self.get_session_info()
        
        # Cleanup
        self.delete_session()
        
        print("\n" + "="*60)
        print("Example Complete!")
        print("="*60)


def example_with_custom_responses():
    """Example showing how to handle different response types."""
    print("\n\nExample 2: Handling Different Response Types")
    print("="*60)
    
    api_url = "http://localhost:8000"
    
    # Create session
    session_res = requests.post(f"{api_url}/sessions")
    session_id = session_res.json()['session_id']
    print(f"Session: {session_id}")
    
    # Send message
    msg = "A blockchain-based supply chain tracking platform"
    msg_res = requests.post(
        f"{api_url}/chat/{session_id}",
        json={"message": msg}
    )
    result = msg_res.json()
    
    # Handle response - could be dict or string
    response = result['response']
    
    if isinstance(response, dict):
        print("\nResponse is structured JSON:")
        for key, value in response.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {str(value)[:100]}...")
    else:
        print(f"\nResponse is text:\n{response}")
    
    # Cleanup
    requests.delete(f"{api_url}/sessions/{session_id}")
    print("\nSession cleaned up")


def example_error_handling():
    """Example showing error handling."""
    print("\n\nExample 3: Error Handling")
    print("="*60)
    
    api_url = "http://localhost:8000"
    
    # Try to send message without session
    print("\n1. Sending message without session (should fail):")
    try:
        response = requests.post(
            f"{api_url}/chat/invalid-session-id",
            json={"message": "test"}
        )
        if response.status_code == 404:
            print(f"   Got expected 404: {response.json()['detail']}")
        else:
            print(f"   Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Try to send empty message
    print("\n2. Creating valid session...")
    session_res = requests.post(f"{api_url}/sessions")
    session_id = session_res.json()['session_id']
    print(f"   Session: {session_id}")
    
    print("\n3. Sending empty message (should fail):")
    try:
        response = requests.post(
            f"{api_url}/chat/{session_id}",
            json={"message": ""}
        )
        if response.status_code == 400:
            print(f"   Got expected 400: {response.json()['detail']}")
        else:
            print(f"   Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Cleanup
    requests.delete(f"{api_url}/sessions/{session_id}")
    print("\n   Session cleaned up")


if __name__ == "__main__":
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Run main example
    example = OriginHubAPIExample(api_url)
    example.run_example()
    
    # Show other examples
    print("\n\nOther examples available:")
    print("  - example_with_custom_responses()")
    print("  - example_error_handling()")
    print("\nRun: python this_file.py to see the main example")
