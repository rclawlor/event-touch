# event-touch
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library containing event-based tactile sensing algorithms.

## Installation
Pending inital release

## Usage
Event data can be simulated from RGB images using the `SimulateEvent` class:
```python
event = SimulateEvent(event_threshold=0.1)
event.initialise_pixel_memory(RGB_frame_0)

event_frame = event.simulate_event(RGB_image_1)
```
