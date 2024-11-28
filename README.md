# Multi-Camera Motion Capture Dataset Collection System

This project aims to develop an end-to-end system for training specialized 3D motion capture models using synchronized multi-camera video. The goal is to create a complete pipeline for collecting training data and developing models for wearable-free motion capture of body, hand, and facial movement.

### Completed Components
- **Multi-Camera Recording System**: Achieved 9μs frame synchronization across cameras using PTP on Raspberry Pis
- **Event-Driven Video Pipeline**: Signal-based architecture achieves thread-level concurrency with microsecond-precision timing
- **Network Streaming**: H.264 encoding and TCP streaming with automatic recovery and segmentation

### In Development
- Stereo camera calibration tools
- Dataset generation using pretrained 2D pose estimation models
- Specialized neural networks for real-time 3D motion capture

The system uses standard hardware and network protocols to achieve high-precision synchronization, enabling accurate 3D reconstruction for generating training data.

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
   - PTP-synchronized system clocks providing sub-microsecond time alignment
   - A signal-based event architecture handling timing-critical operations
   - A semaphore-controlled main loop that sleeps when no work is needed
   - DMA transfers and lock-free queuing ensuring consistent frame timing

3. **Termination**: A "STOP" message ends recording across all cameras simultaneously

A dedicated network handles PTP synchronization, with one Raspberry Pi serving as the grandmaster clock. This precise timing foundation, combined with the event-driven design, enables consistent sub-10μs synchronization despite each camera operating independently.

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

#### Low-Latency Event Handling
While PTP continuously runs in the background maintaining clock synchronization, the recording software focuses purely on frame capture timing. Signal handlers process timing-critical operations, and realtime scheduling ensures these handlers execute immediately when triggered. The process effectively becomes its own scheduler - it gets maximum priority on its dedicated CPU core, and its signal-driven architecture built around a central semaphore means the main loop only activates in response to real events, with zero polling or busy waiting.

The logs demonstrate how these components work together - frame captures align within 9 microseconds across cameras, maintaining precise 33.33ms intervals for 30fps recording. The recording software achieves this by reading from system clocks that PTP keeps tightly synchronized behind the scenes.

### Video Pipeline

The video pipeline is a single-threaded recording application driven by two signal handlers - a design that evolved from multi-threaded versions after finding that cross-core data sharing and additional synchronization complexity reduced performance. When triggered by the timer reaching a target timestamp, one handler queues a capture request to the camera. Upon DMA buffer completion, a second handler enqueues the buffer pointer into a lock-free queue for processing. The main loop dequeues these buffers, performs H.264 encoding, and streams the result via TCP to the server's FFmpeg instance. A semaphore tracking queue size prevents busy waiting in the main loop. The lock-free queue design is crucial, allowing the capture completion handler to safely preempt the main loop while preventing data races between components.

Once the pipeline is running, timing between capture signal, capture completion, and frame transmission shows remarkable consistency, with the system maintaining precise 33.33ms intervals between frames. A timer-based disconnect mechanism closes the TCP connection after 0.3 seconds without new frames, enabling the server to detect recording completion and prepare for the next session. To enable precise timing analysis, a custom async-signal-safe logger captures events directly within signal handlers. Sub-millisecond synchronization is verified through comparing timestamps from these logs across cameras, made possible by the PTP clock synchronization across the capture network.

### Server-Side Pipeline

The server-side pipeline takes advantage of FFmpeg's native support for handling incoming streams, eliminating the need for a conventional server application. The system uses systemd services to handle FFmpeg instances, with a separate service instance listening on a dedicated port for each incoming camera stream. Incoming H.264 streams are recorded using FFmpeg's -segment feature, which creates new files for each segment within a session to allow seamless recovery after interruptions. The streams are transcoded server-side to H.265, leveraging GPU acceleration for efficient compression while minimizing computational load on the Raspberry Pis.

A manager script provides commands to start, stop, restart, and monitor all services. The manager uses a simple counter stored in a dotfile to enumerate recording sessions, ensuring unique filenames across sessions. When launching a new recording session, the manager script broadcasts the initial synchronization timestamp to all cameras via UDP, then starts the FFmpeg services to receive the incoming video streams. The system monitors segment creation to confirm successful recording startup and uses inotify to detect when recording has completed on all cameras.

### Calibration Pipeline
- Lens distortion correction
- Camera alignment
- Stereo calibration
- Frame validation

## Hardware Setup

### Components
- Camera specifications
- Frame construction
- GPIO wiring
- ArUco markers

### Assembly
- Frame assembly
- Camera mounting
- Electronics installation
- Calibration markers

## Dataset Generation

### Pipeline Overview
- Video preprocessing
- MediaPipe integration
- Multi-view triangulation
- Data format

### Output Format
- Dataset structure
- File formats
- Sample counts
- Data fields

## Results & Examples
[Visual examples of system output, calibration results, etc.]

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Any credits or acknowledgments
