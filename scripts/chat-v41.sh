#!/bin/bash
# All-in-one: start the V4.1 server if it isn't running, then open a chat client
# against it in this terminal. Exiting the chat stops the server (unless it was
# already running before this started).
#
#   CHAT_CMD="hermes -p v41 chat" scripts/chat-v41.sh
#
# CHAT_CMD is any OpenAI-compatible chat client pointed at http://127.0.0.1:8002/v1.
PORT=8002; URL="http://127.0.0.1:${PORT}/v1/models"
LOG="${TMPDIR:-/tmp}/ds4-v41-server.log"; SERVER_PID=""
CHAT_CMD="${CHAT_CMD:-hermes -p v41 chat}"
up() { curl -fsS --max-time 2 "$URL" >/dev/null 2>&1; }
cleanup() { [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null && { echo "Stopping V4.1 server..."; kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null; }; }
trap cleanup EXIT INT TERM
if up; then echo "V4.1 server already running -- connecting."
else
  echo "Starting V4.1 server (log: $LOG)..."
  "$(dirname "$0")/serve-v41.sh" > "$LOG" 2>&1 & SERVER_PID=$!
  deadline=$((SECONDS + 600))
  until up; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "Server exited early:"; tail -20 "$LOG"; exit 1; }
    [ $SECONDS -ge $deadline ] && { echo "Timed out waiting for the server."; exit 1; }
    sleep 2
  done
  echo "Server ready."
fi
$CHAT_CMD "$@"
