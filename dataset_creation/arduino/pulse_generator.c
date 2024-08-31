#include <avr/io.h>
#include <avr/interrupt.h>

ISR(TIMER1_COMPA_vect) {
  PORTB ^= _BV(PORTB5);                // Toggle pin 13
  TCCR2B |= (1 << CS21);               // Start timer 2 with prescaler 8
}

ISR(TIMER2_COMPA_vect) {
  TCCR2B &= ~(1 << CS21);              // Stop timer 2
  PORTB ^= _BV(PORTB5);                // Toggle pin 13
}

int main() {
  DDRB |= _BV(DDB5);                   // Set pin 13 as output

  // Configure timer 1 for 30 Hz pulse generation
  TCCR1A = 0;                          // 0 out timer control register A
  TCCR1B = 0;                          // 0 out timer control register B
  TCNT1 = 0;                           // Initialize timer to 0
  OCR1A = 8333;                        // Assign compare value for 30 Hz
  TCCR1B |= (1 << WGM12);              // Clear timer on compare match
  TCCR1B |= (1 << CS11) | (1 << CS10); // Set prescaler to 64
  TIMSK1 |= (1 << OCIE1A);             // Enable compare match interrupt

  // Configure timer 2 for 100 micros pulse width
  TCCR2A = 0;                          // 0 out timer control register A
  TCCR2B = 0;                          // 0 out timer control register B
  TCNT2 = 0;                           // Initialize timer to 0
  OCR2A = 199;                         // Assign compare value for 100 micros
  TCCR2A |= (1 << WGM21);              // Clear timer on compare match
  TIMSK2 |= (1 << OCIE2A);             // Enable compare match interrupt

  sei();                               // Enable global interrupts

  while (1) {
    // Wait for interrupts
  }
}
