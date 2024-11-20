#!/bin/bash
# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

PORTS=(12345 12346 12347)
TIMEOUT=5  # Seconds to wait before considering stream inactive
LOG_FILE="/var/log/ffmpeg_watchdog.log"

log() {
    local level="$1"
    local message="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
}

find_first_segment() {
    local counter=$(cat ".video_counter.txt")
    for port in "${PORTS[@]}"; do
        local pattern="/tmp/port${port}_vid${counter}_*.mp4"
        local segment=$(ls $pattern 2>/dev/null | head -n 1)
        if [ -n "$segment" ]; then
            echo "$segment"
            return 0
        fi
    done
    return 1
}

find_next_segment() {
    local current="$1"
    local base_pattern=$(echo "$current" | sed 's/_[0-9]\+\.mp4$//')
    local current_num=$(echo "$current" | grep -o '_[0-9]\+\.mp4$' | grep -o '[0-9]\+')
    local next_num=$((current_num + 1))
    local next_segment="${base_pattern}_$(printf "%03d" $next_num).mp4"

    if [ -f "$next_segment" ]; then
        echo "$next_segment"
        return 0
    fi
    return 1
}

check_port_activity() {
    local active=false
    for port in "${PORTS[@]}"; do
        if timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
            active=true
            break
        fi
    done
    $active
}

wait_for_initial_segment() {
    log "INFO" "Waiting for initial recording segment..."
    while true; do
        if first_segment=$(find_first_segment); then
            log "INFO" "Found initial segment: $first_segment"
            echo "$first_segment"
            return 0
        fi
        sleep 1
    done
}

monitor_recording() {
    local current_segment="$1"
    log "INFO" "Beginning to monitor recording starting with: $current_segment"

    while true; do
        # Wait for current segment to finalize
        log "DEBUG" "Waiting for segment to finalize: $current_segment"
        if ! inotifywait -e close_write "$current_segment" 2>/dev/null; then
            log "ERROR" "inotifywait failed on: $current_segment"
            return 1
        fi

        # Check for next segment
        if next_segment=$(find_next_segment "$current_segment"); then
            log "DEBUG" "Found next segment: $next_segment"
            current_segment="$next_segment"
            continue
        fi

        # No new segment, check if truly done
        log "INFO" "No new segment found after: $current_segment"
        sleep 1

        if ! next_segment=$(find_next_segment "$current_segment"); then
            # Double check port activity
            if ! check_port_activity; then
                log "INFO" "No port activity detected, waiting $TIMEOUT seconds to confirm..."
                sleep "$TIMEOUT"
                # Final check before declaring recording stopped
                if ! check_port_activity; then
                    log "INFO" "Recording has stopped"
                    return 0
                fi
            fi
        else
            log "DEBUG" "Found delayed next segment: $next_segment"
            current_segment="$next_segment"
            continue
        fi
    done
}


main() {
    log "INFO" "Starting FFmpeg watchdog"

    while true; do
        initial_segment=$(wait_for_initial_segment)
        if [ $? -ne 0 ]; then
            log "ERROR" "Failed to get initial segment"
            sleep 5
            continue
        fi

        if ! monitor_recording "$initial_segment"; then
            log "ERROR" "Error monitoring recording"
            sleep 5
            continue
        fi

        # Recording stopped, restart ffmpeg and launch preprocessing
        log "INFO" "Restarting FFmpeg and launching preprocessing..."
        if ! $(dirname "$0")/ffmpeg_mgr.sh restart; then
            log "ERROR" "Failed to restart FFmpeg"
            sleep 5
            continue
        fi

        # Background the preprocessing
        log "INFO" "Starting preprocessing in background"
        python3 preprocess.py &
        preprocess_pid=$!
        log "DEBUG" "Preprocessing started with PID: $preprocess_pid"
    done
}

main
