import json
import os
import subprocess
import sys


def test_tracing():
    # Payload
    payload = json.dumps({
        "backend": "health", 
        "requests": [],
        "trace_context": {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    })
    
    env = os.environ.copy()
    env["APEX_X_TRACE_EXPORT"] = "true"
    
    # Start service_bridge
    process = subprocess.Popen(
        [sys.executable, "-m", "apex_x.runtime.service_bridge"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    stdout, stderr = process.communicate(input=payload, timeout=5)
    
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    
    if '"name": "process_request"' in stderr or "process_request" in stderr:
        print("SUCCESS: Span 'process_request' found in stderr")
    else:
        print("FAILURE: Span 'process_request' NOT found in stderr")

if __name__ == "__main__":
    test_tracing()
