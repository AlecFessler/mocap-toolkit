# Multi-Camera Hand Pose Dataset Collection System

This is has been my side project for the past few months. Its purpose is to use Google's Mediapipe hand pose predictor, a custom multi-camera enclosure for data capture, and triangulation to bootstrap a 3d and pose dataset at a rate of 108,000 labeled training samples per hour of recording. It employs an avr microcontroller and three raspberry pis to achieve sub millisecond frame capture synchronization and real-time video encoding and streaming to the server where it undergoes a fully scripted preprocessing pipeline. The ultimate goal is to design a neural net architecture to leverage the dataset for accurate, real-time, wearable-free hand motion capture.

### Architecture Diagram

```mermaid
flowchart TD
    classDef lightClass fill:#f5f9ff,color:black,stroke:#d3e3fd
    classDef medClass fill:#d3e3fd,color:black,stroke:#2b6cb0
    classDef darkClass fill:#2b6cb0,color:white
    classDef serverClass fill:#1a4971,color:white

    subgraph MCU["Microcontroller Layer"]
        AVR["AVR Timer\n(30Hz Pulse)"]
        GPIO["GPIO Lines"]
        AVR --> GPIO
    end

    GPIO --> |"Hardware Interrupt"| KM1 & KM2 & KM3

    subgraph RPi1["Raspberry Pi 1"]
        KM1["Kernel Module"] --> Cap1["Camera Handler"] --> Enc1["H.264 Encoder"]
    end

    subgraph RPi2["Raspberry Pi 2"]
        KM2["Kernel Module"] --> Cap2["Camera Handler"] --> Enc2["H.264 Encoder"]
    end

    subgraph RPi3["Raspberry Pi 3"]
        KM3["Kernel Module"] --> Cap3["Camera Handler"] --> Enc3["H.264 Encoder"]
    end

    Enc1 & Enc2 & Enc3 --> |"TCP Stream"| FF

    subgraph Server["Server Layer"]
        FF["FFmpeg Services"] --> Segments["Video Segments"]
        Mgr["Manager Script"] --> WD["Watchdog"]
        WD --> |"Monitor/Control"| FF
        WD --> |"Streaming Done"| Pre["Preprocessing"]
        Segments --> Pre
    end

    class AVR,GPIO lightClass
    class KM1,KM2,KM3,Cap1,Cap2,Cap3 medClass
    class Enc1,Enc2,Enc3,FF darkClass
    class Mgr,WD,Pre,Segments serverClass
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
