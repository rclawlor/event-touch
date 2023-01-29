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
<p align="center">
  <img src="https://github.com/rclawlor/event-touch/blob/master/about_pics/event_generation_digit.png" width="150">
  &emsp;
  <img src="https://github.com/rclawlor/event-touch/blob/master/about_pics/event_generation_memory.png" width="150">
  &emsp;
  <img src="https://github.com/rclawlor/event-touch/blob/master/about_pics/event_generation_frame.png" width="150">
</p>
<p align="center">
  RGB frame
  &emsp;&emsp;&emsp;&emsp;&emsp;
  Pixel memory
  &emsp;&emsp;&emsp;&emsp;&emsp;
  Event frame
</p>
