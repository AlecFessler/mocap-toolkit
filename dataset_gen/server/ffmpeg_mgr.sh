#!/bin/bash
# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

PORTS=(12345 12346 12347)
COUNTER_FILE=".video_counter.txt"
WATCHDOG_PID_FILE="/tmp/ffmpeg_watchdog.pid"
LOG_FILE="/var/log/ffmpeg_mgr.log"

log() {
    local level="$1"
    local message="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
}

if [ ! -f "$COUNTER_FILE" ]; then
    log "INFO" "Initializing counter file"
    echo "0" > "$COUNTER_FILE"
fi

next_counter() {
    counter=$(cat "$COUNTER_FILE")
    next_counter=$((counter + 1))
    echo "$next_counter" > "$COUNTER_FILE"
    log "INFO" "Incremented counter to $next_counter"
    echo "$next_counter"
}

case $1 in
    start)
        log "INFO" "Starting FFmpeg instances..."
        counter=$(next_counter)
        for port in "${PORTS[@]}"; do
            filename="port${port}_vid${counter}"
            log "DEBUG" "Setting up FFmpeg instance for port $port with filename $filename"
            echo "FILENAME=/tmp/port${port}_vid${counter}_%03d.mp4" > /etc/default/ffmpeg-stream
            if ! sudo systemctl start ffmpeg-stream@"$port".service; then
                log "ERROR" "Failed to start ffmpeg-stream@$port.service"
                continue
            fi
            log "INFO" "Started ffmpeg-stream@$port.service with filename $filename"
        done

        log "INFO" "Starting watchdog..."
        ./ffmpeg_watchdog.sh &
        echo $! > "$WATCHDOG_PID_FILE"
        log "INFO" "Started watchdog with PID $(cat $WATCHDOG_PID_FILE)"
        ;;

    stop)
        log "INFO" "Stopping FFmpeg instances..."
        for port in "${PORTS[@]}"; do
            if ! sudo systemctl stop ffmpeg-stream@"$port".service; then
                log "ERROR" "Failed to stop ffmpeg-stream@$port.service"
                continue
            fi
            rm -f "/tmp/recording_active_${port}"
            log "INFO" "Stopped ffmpeg-stream@$port.service"
        done

        if [ -f "$WATCHDOG_PID_FILE" ]; then
            log "INFO" "Stopping watchdog..."
            kill $(cat "$WATCHDOG_PID_FILE")
            rm "$WATCHDOG_PID_FILE"
            log "INFO" "Stopped watchdog"
        fi
        ;;

    status)
        log "INFO" "Checking status of FFmpeg instances..."
        for port in "${PORTS[@]}"; do
            log "DEBUG" "Checking status of ffmpeg-stream@$port.service"
            sudo systemctl status ffmpeg-stream@"$port".service
        done
        ;;

    restart)
        log "INFO" "Restarting FFmpeg instances..."
        counter=$(next_counter)
        for port in "${PORTS[@]}"; do
            filename="port${port}_vid${counter}"
            log "DEBUG" "Setting up FFmpeg instance for port $port with filename $filename"
            echo "FILENAME=/tmp/port${port}_vid${counter}_%03d.mp4" > /etc/default/ffmpeg-stream
            if ! sudo systemctl restart ffmpeg-stream@"$port".service; then
                log "ERROR" "Failed to restart ffmpeg-stream@$port.service"
                continue
            fi
            log "INFO" "Restarted ffmpeg-stream@$port.service with filename $filename"
        done
        ;;

    *)
        log "ERROR" "Invalid command: $1"
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
