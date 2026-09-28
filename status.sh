#!/bin/bash
# Report room-speaker readiness, voice-service state, and PC awake/asleep.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/jarvis.env"   # ROOM_USER ROOM_HOST ROOM_KEY PC_USER PC_HOST PC_KEY
JERRY="ssh -i $ROOM_KEY -o BatchMode=yes -o ConnectTimeout=5 $ROOM_USER@$ROOM_HOST"
if $JERRY "test -x ~/jarvis/say.sh && echo ok" 2>/dev/null | grep -q ok; then room="ready"; else room="down"; fi
svc=$($JERRY "export XDG_RUNTIME_DIR=/run/user/1000; systemctl --user is-active jerry-voice.service" 2>/dev/null | tr -d '\r')
if ssh -i "$PC_KEY" -o BatchMode=yes -o ConnectTimeout=4 "$PC_USER@$PC_HOST" "echo ok" >/dev/null 2>&1; then pc="awake"; else pc="asleep"; fi
echo "{\"room\":\"$room\",\"listening\":\"$svc\",\"pc\":\"$pc\"}"
