# Multi-Camera Motion Capture Toolkit

### Completed Components

### In Development

### System Architecture

The system achieves microsecond-level frame synchronization using a remarkably simple coordination model. A central server communicates with camera processes through just two messages - an initial timestamp broadcast and a stop signal. The power of this minimal design comes from how each camera leverages precise clock synchronization to maintain perfect timing.

The logs tell the story better than words can. Here's a sequence showing the actual synchronization achieved across three cameras recording at 30fps, with timestamps captured independently by each camera:

```
Frame 1:
Camera 1: [2024-11-27 03:36:51.957246] [INFO] Capture request queued
Camera 2: [2024-11-27 03:36:51.957239] [INFO] Capture request queued
Camera 3: [2024-11-27 03:36:51.957239] [INFO] Capture request queued

Frame 2:
Camera 1: [2024-11-27 03:36:51.990584] [INFO] Capture request queued
Camera 2: [2024-11-27 03:36:51.990581] [INFO] Capture request queued
Camera 3: [2024-11-27 03:36:51.990588] [INFO] Capture request queued

Frame 3:
Camera 1: [2024-11-27 03:36:52.023917] [INFO] Capture request queued
Camera 2: [2024-11-27 03:36:52.023923] [INFO] Capture request queued
Camera 3: [2024-11-27 03:36:52.023922] [INFO] Capture request queued
```

The timestamps reveal two key achievements:
1. Frame captures are synchronized within 9 microseconds across cameras
2. The 33.33ms intervals between frames (30fps) are maintained with exceptional precision

The recording lifecycle consists of three phases:

1. **Initialization**:
   - The server broadcasts a single timestamp to all recording processes
   - Each camera independently calculates its entire frame schedule using a simple formula:
     `Tn = timestamp + n * frame_duration`
   - Even if cameras receive the timestamp at slightly different times, they automatically synchronize to this schedule

2. **Recording**: Each camera runs autonomously but in perfect coordination through:

3. **Termination**: A "STOP" message ends recording across all cameras simultaneously

![System Architecture](assets/architecture_diagram.svg)

The architecture diagram illustrates the three Recording & Streaming processes coordinated by the Server Script, with the PTP Grandmaster Clock ensuring precise timing synchronization across all devices. Each process maintains its own capture schedule but remains perfectly aligned through their shared time reference.

### Physical Setup

[Photos/diagrams of recording frame and hardware]

## Technical Components

### Camera Synchronization

The system's tight frame synchronization emerges from the combination of hardware capabilities and network protocols, with PTP synchronization providing the foundation and an event-driven architecture ensuring precise frame timing.

#### Precision Time Protocol (PTP)
The Raspberry Pi 5's ethernet ports include dedicated hardware timestamping support, enabling PTP to achieve sub-microsecond clock synchronization across the camera network. While NTP typically achieves millisecond-level synchronization, PTP takes advantage of hardware timestamping to measure and compensate for network delays with much greater precision, making it ideal for creating a shared time reference across cameras.

#### The Problem with Real Time
Trying to capture frames at specific timestamps using the system clock (CLOCK_REALTIME) leads to a fatal flaw: while the clock steadily moves forward in time, PTP is constantly making tiny adjustments forward and backward to keep it synchronized. This means when you target a specific nanosecond timestamp, that exact number might literally never appear in the clock's sequence of values. The clock could jump right over it during an adjustment, going from slightly before your target to slightly after it. This isn't a theoretical problem - it caused a complete deadlock every single time before switching to monotonic clock timing. The process would get stuck waiting on the semaphore forever because it was waiting for a precise nanosecond value that the realtime clock would simply never reach.

#### The Solution: Monotonic Timing
The system resolves this by using the monotonic clock as a stable reference point:

1. When a camera receives the initial timestamp, it samples both clocks in rapid succession:
   ```cpp
   clock_gettime(CLOCK_REALTIME, &real_time);    // PTP-synchronized time
   clock_gettime(CLOCK_MONOTONIC, &mono_time);   // Steadily increasing time
   ```

2. It calculates how far in the future the target timestamp is relative to the current realtime clock:
   ```cpp
   int64_t ns_until_target = timestamp - current_real_ns;
   ```

3. This same time delta is then applied to the monotonic clock reading to get an absolute monotonic target:
   ```cpp
   int64_t mono_target_ns = current_mono_ns + ns_until_target;
   ```

4. The timer is set to this absolute monotonic target using TIMER_ABSTIME:
   ```cpp
   timer_settime(timerid, TIMER_ABSTIME, &its, NULL);
   ```

Setting absolute timestamps in the monotonic clock domain ensures each timer will trigger at the exact intended moment, regardless of how long the setup takes or what the current time is. This preserves the synchronization established by the PTP-synced realtime clock while providing a stable time reference that won't be adjusted.

Even when cameras receive the initial timestamp at slightly different times, they can calculate the correct monotonic clock targets for future frames. The system compensates by detecting if a target timestamp is in the past and adjusting forward by the appropriate number of frame intervals.

### Video Pipeline

### Server-Side Pipeline

### Calibration Pipeline

## Hardware Setup

## Dataset Generation

### Pipeline Overview

## Results & Examples
[Visual examples of system output, calibration results, etc.]

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Any credits or acknowledgments
