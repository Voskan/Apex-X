import sys

from apex_x.utils.logging import get_logger, log_event

logger = get_logger("test_logging")

print("--- Test 1: Info Log ---", file=sys.stderr)
logger.info("Hello JSON!", key="value", status=200)

print("\n--- Test 2: Error Log with Traceback ---", file=sys.stderr)
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("Something went wrong", error_code="DIV_ZERO")

print("\n--- Test 3: Legacy log_event ---", file=sys.stderr)
log_event(logger, "legacy_event", fields={"old": "school"})
