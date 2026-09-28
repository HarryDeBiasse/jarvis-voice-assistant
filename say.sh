#!/bin/bash
# Speak text on the room speaker (Pi) or on the PC (Windows SAPI, Wake-on-LAN if asleep).
# Real hosts/keys/MAC live in jarvis.env (gitignored). See jarvis.env.example.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/jarvis.env"   # ROOM_USER ROOM_HOST ROOM_KEY PC_USER PC_HOST PC_KEY PC_MAC LAN_BROADCAST
TGT="${1:-room}"; B64="$2"; VK="${3:-David}"; VOL="${4:-85}"; RATE="${5:-0}"
if [ "$TGT" = "room" ]; then
  # rate -> piper length_scale (higher = slower)
  LS=$(awk "BEGIN{printf \"%.2f\", 1.0 - ($RATE*0.06)}")
  JERRY="ssh -i $ROOM_KEY -o BatchMode=yes -o ConnectTimeout=8 $ROOM_USER@$ROOM_HOST"
  res=$($JERRY "~/jarvis/say.sh '$B64' $LS" 2>&1 | tr -d '\r')
  echo "$res" | grep -q DONE && echo '{"ok":true,"via":"room"}' || echo '{"ok":false,"error":"room_speak_failed"}'
  exit 0
fi
# pc target: SAPI on the PC, WoL if asleep
case "$VK" in Zira) VN="Microsoft Zira Desktop";; *) VN="Microsoft David Desktop";; esac
woke="no"
if ! ssh -i "$PC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$PC_USER@$PC_HOST" "echo ok" >/dev/null 2>&1; then
  woke="yes"; wakeonlan "$PC_MAC" >/dev/null 2>&1; wakeonlan -i "$LAN_BROADCAST" "$PC_MAC" >/dev/null 2>&1
  ok=""; for i in $(seq 1 25); do ssh -i "$PC_KEY" -o BatchMode=yes -o ConnectTimeout=4 "$PC_USER@$PC_HOST" "echo ok" >/dev/null 2>&1 && { ok=1; break; }; sleep 3; done
  [ -z "$ok" ] && { echo '{"ok":false,"error":"pc_wake_timeout"}'; exit 0; }
fi
res=$(ssh -i "$PC_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$PC_USER@$PC_HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\Temp\\jarvis_say.ps1 -b64 $B64 -voice \"$VN\" -vol $VOL -rate $RATE" 2>&1 | tr -d '\r')
echo "$res" | grep -q DONE && echo "{\"ok\":true,\"via\":\"pc\",\"woke\":\"$woke\"}" || echo '{"ok":false,"error":"speak_failed"}'
