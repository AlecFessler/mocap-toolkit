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

The video pipeline achieves true concurrency through a signal-driven architecture built around three core components that drive the recording cycle forward: a timer signal handler for initiating captures, a camera completion callback for queueing frames, and a main loop for frame processing. This design evolved from traditional multi-threaded approaches after finding that the overhead of cross-core data sharing and additional synchronization complexity actually reduced performance.

#### Signal-Driven Recording Cycle
When a timestamp is present, the system drives itself forward through a continuous cycle. At the start of each iteration, the main loop calculates the next target timestamp and arms the timer, ensuring the cycle will continue. The loop then blocks on the semaphore, knowing with certainty that either a timer signal will arrive or a frame will become available for processing.

The cycle flows naturally:
1. At the precise moment specified by the timer, the signal handler fires and queues a capture request to the camera
2. When capture completes, the camera's callback enqueues the frame to a lock-free queue and increments the semaphore
3. The main loop unblocks and processes the frame, encoding and streaming it
4. The loop calculates the next timestamp and arms the timer before blocking again

This cycle continues indefinitely until the server sends a "STOP" message, which simply unsets the timestamp. At this point, no new timers will be armed, but the semaphore's count ensures the main loop continues processing any remaining frames before exiting, providing clean shutdown without data loss.

#### Concurrency Without Threads
The system achieves true concurrency but without the complexity of threading. For example, after queueing a capture request, the main loop can process any backlog of frames while the camera is capturing the next image. This parallelizes I/O with CPU operations just like threading would, but without the overhead of context switching.

The lock-free queue enables safe concurrent access between the camera's callback and the main loop. The callback can safely preempt the main loop even during a dequeue operation - when the callback returns, the main loop's compare-and-swap operation detects the interruption and retries. This achieves concurrent access with minimal overhead in a completely single-threaded design.

DMA transfers ensure zero-copy frame capture, with the camera writing directly to memory buffers that are then passed through the pipeline via pointers in the lock-free queue. This minimizes both latency and memory usage, as frames never need to be copied between pipeline stages.

#### Precision Through Scheduling
The process runs with maximum priority FIFO scheduling on a dedicated CPU core. This means any process of equal priority must wait until this one is blocking on the semaphore before it can be scheduled on our core. Additionally, any process of lower priority will be preempted as soon as we have a signal to handle or the semaphore is unblocked.

The significance of this scheduling becomes clear in the pipeline's operation. Timing-critical operations are handled by signal handlers, ensuring immediate response to timer events and camera callbacks. Less timing-critical operations like encoding and streaming occur in the main loop. This natural separation, combined with the FIFO scheduling, means the process effectively becomes its own scheduler.

A custom logger enables precise timestamp logging directly within signal handlers through atomic file writes and entirely reentrant string formatting. This avoids issues with non-reentrant functions like sprintf while providing the precise timing data seen in the logs, verifying both the frame synchronization and the low-latency signal handling.

#### Efficient Control Flow
The main loop's design around a single semaphore creates remarkably efficient control flow. The semaphore count exactly matches the number of frames available for processing, ensuring the loop only iterates when there is real work to do. When no frames are available, the process sleeps, consuming no CPU cycles until the next frame capture completes.

This efficiency makes multi-core parallelization unnecessary - the process spends so little time active that the overhead of cross-core data transfer would outweigh any potential benefits. The single-core, event-driven design provides all the concurrency needed while maintaining precise timing control. In practice, frame processing typically completes within 5ms of the initial capture signal, leaving plenty of headroom within the 33.33ms frame interval.

The consistent timing between capture signal, capture completion, and frame transmission shown in the logs verifies this efficiency. Each step in the pipeline executes with predictable latency, maintaining the precise 33.33ms intervals required for 30fps video while using minimal system resources.

### Server-Side Pipeline

The server-side pipeline leverages FFmpeg's native support for handling incoming streams, eliminating the need for a conventional server application. The system uses a systemd service template to manage independent FFmpeg instances, with each instance listening on a dedicated port for its corresponding camera stream. This design achieves robust multi-camera recording through remarkably simple means.

#### Recording Management

The recording lifecycle begins when the manager script broadcasts an initial synchronization timestamp to all cameras via UDP. The script then launches FFmpeg services to receive the incoming video streams. Each stream undergoes GPU-accelerated transcoding from H.264 to H.265, optimizing storage while minimizing computational load on the Raspberry Pis.

FFmpeg's segment feature automatically divides recordings into sequential chunks, creating new files for each segment within a session. This segmentation strategy provides natural recovery points after any interruptions while maintaining continuous recording. The system monitors segment creation to confirm successful recording startup and uses inotify to detect when recording has completed on all cameras.

#### Session Management

A simple but effective session tracking system uses an atomic counter stored in a dotfile to enumerate recording sessions. This ensures unique filenames across sessions while maintaining a clear organizational structure. The counter automatically increments with each new recording, generating paths that clearly identify the source camera, session number, and segment sequence.

The entire server implementation achieves sophisticated functionality through the strategic combination of standard system tools. By leveraging FFmpeg's capabilities and systemd's service management, the system delivers reliable multi-camera streaming with minimal custom code and maximum resilience.

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
