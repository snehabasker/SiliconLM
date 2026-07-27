# DemoChip DC-100 Datasheet (sample/fictional part, for RAG demo purposes)

## 1. Overview
The DC-100 is a fictional 8-bit microcontroller used here only to give the
SiliconLM RAG demo something concrete to retrieve and cite. It is not a real
commercial part.

## 2. Absolute Maximum Ratings
- Supply voltage (VDD): -0.3V to 4.0V. Exceeding 4.0V may cause permanent
  damage to the device.
- Input voltage on any pin: -0.3V to VDD + 0.3V.
- Storage temperature range: -65C to +150C.
- Junction temperature (TJ max): 150C.
- ESD (Human Body Model): 2000V on all pins.

## 3. Recommended Operating Conditions
- Supply voltage (VDD): 1.8V to 3.6V, nominal 3.3V.
- Operating ambient temperature: -40C to +85C (industrial grade).
- Maximum clock frequency at 3.3V: 48 MHz.

## 4. Power Consumption
- Active mode at 48 MHz, 3.3V: 12 mA typical.
- Sleep mode: 2 uA typical.
- Deep sleep with RTC running: 0.6 uA typical.

## 5. Reset Behavior
The DC-100 holds all outputs in high-impedance state until VDD crosses the
power-on-reset threshold of 1.5V, then releases reset after a 10ms internal
delay to allow the oscillator to stabilize.
