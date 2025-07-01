#!/usr/bin/env python3
"""Simple test to verify Range structure is working in HTML output."""

import tracerite
from tracerite.html import html_traceback


def test_function():
    x = 42
    y = "hello"
    result = x + y  # This will cause a TypeError


if __name__ == "__main__":
    try:
        test_function()
    except Exception:
        # Extract frames to verify Range structure
        frames = tracerite.extract_chain()[-1]["frames"]
        print("Frame structure:")
        for frame in frames:
            print(f"  Function: {frame['function']}")
            print(f"  Range: {frame['range']}")
            if frame["range"]:
                print(f"    Lines: {frame['range'].lfirst} to {frame['range'].lfinal}")
                print(f"    Columns: {frame['range'].cbeg} to {frame['range'].cend}")
            print()

        # Generate HTML to verify it works
        html = html_traceback()
        print("HTML generation successful!")
        print(f"HTML length: {len(str(html))} characters")
