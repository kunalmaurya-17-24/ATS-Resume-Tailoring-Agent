from agent.graph import create_resume_agent
import os

import requests
import base64

def visualize():
    try:
        agent = create_resume_agent()
        mermaid_text = agent.get_graph().draw_mermaid()
        print("Mermaid text generated.")
        
        # Use mermaid.ink to generate a PNG
        # Convert text to base64
        base64_str = base64.b64encode(mermaid_text.encode('utf-8')).decode('utf-8')
        url = f"https://mermaid.ink/img/{base64_str}"
        
        print(f"Downloading image from: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open("langgraph_viz.png", "wb") as f:
                f.write(response.content)
            print("Success: Graph visualization saved as 'langgraph_viz.png'")
        else:
            print(f"Failed to download image: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    visualize()
