#!/usr/bin/env python3
"""
Test script to test litellm.completion with specific parameters
"""

import json
import litellm


def test_llm_completion():
    """Test litellm.completion with the provided parameters"""

    # Load test parameters
    params = json.load(open('data.json'))

    # Set up litellm config for qwen3-coder-480b from config.toml
    params['model'] = "litellm_proxy/qwen3-coder-480b"
    params['api_key'] = "sk-wQknD-CB6AbHdA-tp0vfQg"
    params['base_url'] = "https://llm-proxy.eval.all-hands.dev"
    params['temperature'] = 0.0
    params['max_tokens'] = 8192

    try:
        print("Testing litellm.completion with provided parameters...")
        print(f"Model: {params['model']}")
        print(f"Base URL: {params['base_url']}")

        # Test the completion call (equivalent to agent.llm.completion on line 215)
        response = litellm.completion(**params)

        print(f"Response received: {response}")
        print(f"Response type: {type(response)}")

        if hasattr(response, 'choices') and response.choices:
            print(f"First choice content: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error during litellm.completion call: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_llm_completion()
