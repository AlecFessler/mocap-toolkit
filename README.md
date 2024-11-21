# Multi-Camera Hand Pose Dataset Collection System

This is has been my side project for the past few months. Its purpose is to use Google's Mediapipe hand pose predictor, a custom multi-camera enclosure for data capture, and triangulation to bootstrap a 3d and pose dataset at a rate of 108,000 labeled training samples per hour of recording. It employs an avr microcontroller and three raspberry pis to achieve sub millisecond frame capture synchronization and real-time video encoding and streaming to the server where it undergoes a fully scripted preprocessing pipeline. The ultimate goal is to design a neural net architecture to leverage the dataset for accurate, real-time, wearable-free hand motion capture.

### Architecture Diagram

```mermaid
flowchart TB
    subgraph MCU["Microcontroller Layer"]
        AVR["AVR Timer\n(30Hz Pulse)"]
        GPIO["GPIO Lines"]
        AVR --> GPIO
    end

    subgraph RPi["Raspberry Pi Layer"]
        direction TB
        subgraph Processing1["Processing Pipeline 1"]
            KM1["Kernel Module"]
            Cap1["Camera Handler"]
            Enc1["H.264 Encoder"]
            KM1 --> Cap1
            Cap1 --> Enc1
        end

        subgraph Processing2["Processing Pipeline 2"]
            KM2["Kernel Module"]
            Cap2["Camera Handler"]
            Enc2["H.264 Encoder"]
            KM2 --> Cap2
            Cap2 --> Enc2
        end

        subgraph Processing3["Processing Pipeline 3"]
            KM3["Kernel Module"]
            Cap3["Camera Handler"]
            Enc3["H.264 Encoder"]
            KM3 --> Cap3
            Cap3 --> Enc3
        end
    end

    subgraph Server["Server Layer"]
        FF1["FFmpeg Service 1"]
        FF2["FFmpeg Service 2"]
        FF3["FFmpeg Service 3"]
        Segments["Video Segments"]
        Mgr["Manager Script"]
        WD["Watchdog"]
        Pre["Preprocessing Pipeline"]

        Mgr --> WD
        WD --> |"Monitor/Control"| FF1
        WD --> |"Monitor/Control"| FF2
        WD --> |"Monitor/Control"| FF3
        FF1 --> Segments
        FF2 --> Segments
        FF3 --> Segments
        WD --> |"Streaming Done"| Pre
        Segments --> Pre
    end

    GPIO --> |"Hardware Interrupt"| KM1
    GPIO --> |"Hardware Interrupt"| KM2
    GPIO --> |"Hardware Interrupt"| KM3

    Enc1 --> |"TCP Stream"| FF1
    Enc2 --> |"TCP Stream"| FF2
    Enc3 --> |"TCP Stream"| FF3
```

### Physical Setup
[Photos/diagrams of recording frame and hardware]

## Technical Components

### Camera Synchronization
- GPIO-based hardware sync
- Kernel module
- Real-time scheduling
- Performance characteristics

### Video Pipeline
- Camera capture system
- Real-time encoding
- Network streaming
- File management

### System Management
- Service orchestration
- Process monitoring
- Error handling
- Recovery mechanisms

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
License information

## Acknowledgments
Any credits or acknowledgments
