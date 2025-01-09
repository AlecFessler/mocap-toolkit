# Multi-Camera Motion Capture Toolkit

### Completed Components
- Recording software for raspberry pi 5s capable of synchronizing frame captures across a local network of cameras with microsecond precision
- A server which manages incoming video streams and bundles framesets with aligned timestamps for downstream tools to consume
- A simple-to-use lens calibration tool built on top of the streaming infrastructure, taking only a few seconds to calibrate a camera
- A stereo calibration tool, also built on top of the streaming infrastructure, which calibrates a full camera array at once

### In Development
- A tool to take framesets from stereo calibrated cameras and Meta's Sapiens pose predictor to triangulate the 3D pose coordinates
- A neural network architecture to train on the resulting dataset

### System Architecture

### Physical Setup

[Photos/diagrams of recording frame and hardware]

## Technical Components

### Camera Synchronization

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
1. Frame captures are synchronized within 10 microseconds across cameras
2. The 33.33ms intervals between frames (30fps) are maintained with exceptional precision

The recording lifecycle consists of three phases:

1. **Initialization**:
   - The server broadcasts a single timestamp to all recording processes
   - Each camera independently calculates its entire frame schedule using a simple formula:
     `Tn = timestamp + n * frame_duration`
   - Even if cameras receive the timestamp at slightly different times, they automatically synchronize to this schedule

2. **Recording**: Each camera runs autonomously but in perfect coordination through its timer-driven capture schedule, maintaining synchronization without any further network communication

3. **Termination**: A "STOP" broadcast ends recording across all cameras simultaneously

The system's tight frame synchronization emerges from the combination of hardware capabilities and network protocols, with PTP synchronization providing the foundation and an event-driven architecture ensuring precise frame timing.

#### Precision Time Protocol (PTP)
The Raspberry Pi 5's ethernet ports include dedicated hardware timestamping support, enabling PTP to achieve sub-microsecond clock synchronization across the camera network. While NTP typically achieves millisecond-level synchronization, PTP takes advantage of hardware timestamping to measure and compensate for network delays with much greater precision, making it ideal for creating a shared time reference across cameras.

#### The Problem with Real Time
Trying to capture frames at specific timestamps using the system clock (CLOCK_REALTIME) leads to a fatal flaw: while the clock steadily moves forward in time, PTP is constantly making tiny adjustments forward and backward to keep it synchronized. This means when you target a specific nanosecond timestamp, that exact number might literally never appear in the clock's sequence of values. The clock could jump right over it during an adjustment, going from slightly before your target to slightly after it. This isn't a theoretical problem - it caused a complete deadlock every single time before switching to monotonic clock timing. The process would get stuck waiting on the semaphore forever because it was waiting for a precise nanosecond value that the realtime clock would simply never reach.

#### The Solution: Monotonic Timing
The system converts between realtime and monotonic clock domains to ensure precise timing. For each frame, it:

1. Calculates the target capture time using a simple formula:
   ```cpp
   auto target = initial_timestamp + (counter * interval);
   ```
   This gives us an absolute timestamp in the realtime clock domain, which is synchronized across cameras via PTP. However, we can't directly use this timestamp due to PTP's continuous clock adjustments.

2. Instead, we calculate how far in the future this target is relative to the current realtime clock, and apply that same offset to the monotonic clock:
   ```cpp
   // Sample both clocks
   clock_gettime(CLOCK_REALTIME, &realtime);     // PTP-synchronized time
   clock_gettime(CLOCK_MONOTONIC, &monotime);    // Steadily increasing time

   // Convert time delta from realtime to monotonic domain
   auto ns_til_target = target - real_ns;
   auto mono_target_ns = mono_ns + ns_til_target;
   ```

3. If we've fallen behind schedule (ns_til_target is negative), we adjust forward by the necessary number of intervals:
   ```cpp
   if (ns_til_target <= std::chrono::nanoseconds{0}) {
       uint32_t intervals_elapsed = -ns_til_target / interval + 1;
       ns_til_target += intervals_elapsed * interval;
       counter += intervals_elapsed;
       target += intervals_elapsed * interval;
   }
   ```

4. Finally, we set an absolute timer in the monotonic clock domain:
   ```cpp
   struct itimerspec its;
   its.it_value.tv_sec = std::chrono::duration_cast<std::chrono::seconds>(mono_target_ns).count();
   its.it_value.tv_nsec = (mono_target_ns - std::chrono::seconds{its.it_value.tv_sec}).count();
   timer_settime(timerid, TIMER_ABSTIME, &its, nullptr);
   ```

This approach preserves the precise synchronization established by PTP while avoiding the problems that would arise from using the realtime clock directly for timer scheduling.

Setting absolute timestamps in the monotonic clock domain ensures each timer will trigger at the exact intended moment, regardless of how long the setup takes or what the current time is. This preserves the synchronization established by the PTP-synced realtime clock while providing a stable time reference that won't be adjusted.

Even when cameras receive the initial timestamp at slightly different times, they can calculate the correct monotonic clock targets for future frames. The system compensates by detecting if a target timestamp is in the past and adjusting forward by the appropriate number of frame intervals.

### Video Pipeline

### Server-Side Pipeline

### Calibration Pipeline

### Dataset Generation Pipeline

## Results & Examples
[Visual examples of system output, calibration results, etc.]

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Any credits or acknowledgments
