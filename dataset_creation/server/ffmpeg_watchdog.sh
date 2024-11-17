#!/bin/bash

PORTS=(12345 12346 12347)
TIMEOUT=5 # seconds to wait before considering a stream inactive

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
            echo "All streams inactive, restarting recording..."
            $(dirname "$0")/control_script.sh restart
            # Invoke preprocessing right here
            # python3 preprocess.py &
        fi
    fi
    sleep 1
done
