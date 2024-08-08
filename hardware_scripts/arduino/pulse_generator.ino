#include <avr/io.h>
#include <avr/interrupt.h>

const int pulsePin = 13;  // Pin to output the pulse signal (also the onboard LED pin)

void setup() {
  // Initialize the pulse pin as an output
  pinMode(pulsePin, OUTPUT);
  digitalWrite(pulsePin, LOW);
  noInterrupts();           // Disable all interrupts

  // Configure Timer1 for 30fps interval
  TCCR1A = 0;               // Clear Timer/Counter Control Registers
  TCCR1B = 0;
  TCNT1 = 0;                // Initialize counter value to 0
  // Set compare match register for 33.33ms interval (30fps)
  OCR1A = 8333;             // Set the compare match register for 30fps
  // Turn on CTC mode (Clear Timer on Compare Match)
  TCCR1B |= (1 << WGM12);
  // Set CS11 and CS10 bits for 64 prescaler
  TCCR1B |= (1 << CS11) | (1 << CS10);
  // Enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);

  // Configure Timer2 for pulse width
  TCCR2A = 0;               // Clear Timer/Counter Control Registers
  TCCR2B = 0;
  TCNT2 = 0;                // Initialize counter value to 0
  // Set compare match register for 100 microseconds (assuming 16 MHz clock and 8 prescaler)
  OCR2A = 199;              // (16MHz / 8) * 100e-6 - 1 = 200 - 1 = 199
  // Turn on CTC mode (Clear Timer on Compare Match)
  TCCR2A |= (1 << WGM21);
  // Set CS21 bit for 8 prescaler
  TCCR2B |= (1 << CS21);
  // Enable timer compare interrupt
  TIMSK2 |= (1 << OCIE2A);

  interrupts();             // Enable all interrupts
}

ISR(TIMER1_COMPA_vect) {
  // This function will be called every 33.33ms (for 30fps)
  digitalWrite(pulsePin, HIGH);  // Generate a pulse
  TCNT2 = 0;                     // Reset Timer2 counter
  TCCR2B |= (1 << CS21);         // Start Timer2 with 8 prescaler
}

ISR(TIMER2_COMPA_vect) {
  digitalWrite(pulsePin, LOW);   // End the pulse
  TCCR2B &= ~(1 << CS21);        // Stop Timer2
}

void loop() {
  // Main loop does nothing, everything is handled by the ISRs
}
