#!/bin/bash
# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

PORTS=(12345 12346 12347)
TIMEOUT=5  # Seconds to wait before considering stream inactive

wait_for_recording() {
    local port=$1
    local counter=$2
    local filename="tmp/port${port}_vid${counter}_000.mp4"

    echo "Waiting for recording to start on port ${port}..."
    while true; do
        if [ -f "$filename" ]; then
            local size1=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename")
            sleep 1
            local size2=$(stat -f%z "$filename" 2>/dev/null || stat -c%s "$filename")
            if [ "$size2" -gt "$size1" ]; then
                touch "/tmp/recording_active_${port}"
                echo "Recording detected on port ${port}"
                return 0
            fi
        fi
        sleep 1
    done
}

get_latest_segment() {
    local port=$1
    local counter=$2
    ls -v tmp/port${port}_vid${counter}_*.mp4 2>/dev/null | tail -n 1
}

wait_for_ffmpeg_completion() {
    local counter=$1
    echo "Waiting for FFmpeg to finish writing all segments..."

    for port in "${PORTS[@]}"; do
        latest_file=$(get_latest_segment "$port" "$counter")
        if [ -n "$latest_file" ]; then
            echo "Waiting for $latest_file to complete..."
            inotifywait -e close_write "$latest_file"
            echo "File $latest_file completed"
        else
            echo "No files found for port $port"
        fi
    done

    echo "All segments complete, ready for preprocessing"
}

check_connection() {
    local port=$1
    timeout 1 bash -c "echo >/dev/tcp/localhost/$port" >/dev/null 2>&1
    return $?
}

while true; do
    if ls /tmp/recording_active_* 1>/dev/null 2>&1; then
        all_inactive=true
        for port in "${PORTS[@]}"; do
            if [ -f "/tmp/recording_active_${port}" ]; then
                if ! check_connection "$port"; then
                    # Port not responding, wait TIMEOUT seconds
                    sleep "$TIMEOUT"
                    # Check again
                    if [ -f "/tmp/recording_active_${port}" ] && ! check_connection "$port"; then
                        echo "Port $port inactive for $TIMEOUT seconds"
                    else
                        all_inactive=false
                    fi
                else
                    all_inactive=false
                fi
            fi
        done

        if [ "$all_inactive" = true ]; then
            current_counter=$(cat ".video_counter.txt")
            echo "All streams inactive, waiting for FFmpeg to finish..."
            wait_for_ffmpeg_completion "$current_counter"

            echo "Starting preprocessing..."
            #python3 preprocess.py

            echo "Restarting recording..."
            $(dirname "$0")/control_script.sh restart
        fi
    else
        # Wait for initial recordings on all ports
        counter=$(cat ".video_counter.txt")
        for port in "${PORTS[@]}"; do
            wait_for_recording "$port" "$counter"
        done
    fi
    sleep 1
done
