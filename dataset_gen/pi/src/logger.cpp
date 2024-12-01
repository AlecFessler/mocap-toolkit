// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "logger.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

logger_t::logger_t(const char* filename) {
  /**
   * Creates an async-signal-safe logger that writes to a specified file.
   *
   * Opens a file descriptor with three key flags:
   * - O_WRONLY: Write-only access is all thats needed
   * - O_CREAT:  Creates the file if it doesn't exist
   * - O_APPEND: Atomic append operations prevent log corruption
   *
   * The 0644 permissions allow:
   * - Owner: read/write
   * - Group: read
   * - Others: read
   *
   * Parameters:
   *   filename: Path to the log file
   *
   * Throws:
   *   std::runtime_error if file cannot be opened
   */
  fd_ = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd_ < 0)
    throw std::runtime_error("Failed to open log file");
}

logger_t::~logger_t() {
  /**
   * Cleans up the log file descriptor
   */
  if (fd_ >= 0) close(fd_);
}

static const char* level_to_str(logger_t::level_t level) {
  /**
   * Converts log level enumeration to human-readable string.
   *
   * Uses a constexpr array and switch statement rather than a map
   * to maintain async-signal-safety. The array is initialized at
   * compile time and the switch generates constant-time lookups.
   *
   * Parameters:
   *   level: The log level to convert
   *
   * Returns:
   *   Corresponding string representation, or "UNKNOWN" for invalid levels
   */
  constexpr const char* levels[] = {"INFO", "DEBUG", "WARNING", "ERROR", "UNKNOWN"};
  switch (level) {
    case logger_t::level_t::INFO: return levels[0];
    case logger_t::level_t::DEBUG: return levels[1];
    case logger_t::level_t::WARNING: return levels[2];
    case logger_t::level_t::ERROR: return levels[3];
    default: return levels[4];
  }
}

static void i_to_str(int value, char* buffer, int& offset) {
  /**
   * Converts integers to strings without using any unsafe functions.
   *
   * This function is crucial for async-signal-safety as standard
   * string conversion functions like sprintf or std::to_string
   * are unsafe in signal handlers. The implementation:
   *
   * 1. Handles special case of zero
   * 2. Manages negative numbers by prepending '-'
   * 3. Builds digits in reverse order in a temporary buffer
   * 4. Copies digits to final position in correct order
   *
   * The algorithm avoids:
   * - Dynamic memory allocation
   * - Function calls that might allocate
   * - Global state modifications
   *
   * Parameters:
   *   value:  Integer to convert
   *   buffer: Target character buffer
   *   offset: Current position in buffer, updated as digits are added
   *
   * Note: Caller must ensure buffer has sufficient space for maximum
   * possible number of digits plus sign (11 chars for 32-bit int)
   */
  if (value == 0) {
    buffer[offset++] = '0';
    return;
  }

  if (value < 0) {
    buffer[offset++] = '-';
    value = -value;
  }

  char temp[16];
  int len = 0;
  while (value > 0) {
    temp[len++] = '0' + value % 10;
    value /= 10;
  }

  while (len > 0) {
    buffer[offset++] = temp[--len];
  }
}

#define LEAP_YEAR(year) ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)
constexpr int SECONDS_PER_LEAP_YEAR = 31622400;
constexpr int SECONDS_PER_YEAR = 31536000;
constexpr int SECONDS_PER_MONTH[] = {
  2505600,  // February leap
  2678400,  // January
  2419200,  // February
  2678400,  // March
  2592000,  // April
  2678400,  // May
  2592000,  // June
  2678400,  // July
  2678400,  // August
  2592000,  // September
  2678400,  // October
  2592000,  // November
  2678400   // December
};
constexpr int SECONDS_PER_DAY = 86400;
constexpr int SECONDS_PER_HOUR = 3600;
constexpr int SECONDS_PER_MINUTE = 60;
constexpr int NANOS_PER_MILLISECOND = 1000000;
constexpr int NANOS_PER_MICROSECOND = 1000;

static void timestamp(char* buffer, int& offset) {
  /**
   * Generates an ISO 8601 timestamp without using unsafe time functions.
   *
   * Most time functions (like strftime) are unsafe in signal handlers.
   * This function performs manual time calculations to decompose a
   * CLOCK_REALTIME timestamp into its components:
   *
   * 1. Gets raw seconds and nanoseconds
   * 2. Accounts for leap years since epoch
   * 3. Breaks down into calendar components:
   *    - Years since 1970
   *    - Months (handling February in leap years)
   *    - Days
   *    - Hours, minutes, seconds
   *    - Milliseconds and microseconds
   *
   * Format: "YYYY-MM-DD HH:MM:SS.mmmuuuZ"
   * Example: "2024-03-27 14:30:15.123456Z"
   *
   * Parameters:
   *   buffer: Character buffer for timestamp (needs 28 bytes)
   *   offset: Current position in buffer, updated as timestamp is written
   *
   * Note: The Z suffix indicates UTC timezone, which is what
   * CLOCK_REALTIME provides on Linux systems
   */
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  time_t seconds = ts.tv_sec;
  int nanos = ts.tv_nsec;

  int year = 1970;
  while (true) {
    int seconds_in_year = LEAP_YEAR(year) ?
                        SECONDS_PER_LEAP_YEAR :
                        SECONDS_PER_YEAR;
    if (seconds < seconds_in_year)
      break;
    seconds -= seconds_in_year;
    year++;
  }

  int month = 1;
  while (true) {
    int seconds_in_month = LEAP_YEAR(year) && month == 2 ?
                         SECONDS_PER_MONTH[0] :
                         SECONDS_PER_MONTH[month];
    if (seconds < seconds_in_month)
      break;
    seconds -= seconds_in_month;
    month++;
  }

  int day = seconds / SECONDS_PER_DAY + 1;
  seconds %= SECONDS_PER_DAY;
  int hour = seconds / SECONDS_PER_HOUR;
  seconds %= SECONDS_PER_HOUR;
  int minute = seconds / SECONDS_PER_MINUTE;
  seconds %= SECONDS_PER_MINUTE;
  int millis = nanos / NANOS_PER_MILLISECOND;
  nanos %= NANOS_PER_MILLISECOND;
  int micros = nanos / NANOS_PER_MICROSECOND;

  i_to_str(year, buffer, offset);
  buffer[offset++] = '-';
  if (month < 10)
    buffer[offset++] = '0';
  i_to_str(month, buffer, offset);
  buffer[offset++] = '-';
  if (day < 10)
    buffer[offset++] = '0';
  i_to_str(day, buffer, offset);
  buffer[offset++] = ' ';
  if (hour < 10)
    buffer[offset++] = '0';
  i_to_str(hour, buffer, offset);
  buffer[offset++] = ':';
  if (minute < 10)
    buffer[offset++] = '0';
  i_to_str(minute, buffer, offset);
  buffer[offset++] = ':';
  if (seconds < 10)
    buffer[offset++] = '0';
  i_to_str(seconds, buffer, offset);
  buffer[offset++] = '.';
  if (millis < 100)
    buffer[offset++] = '0';
  if (millis < 10)
    buffer[offset++] = '0';
  i_to_str(millis, buffer, offset);
  if (micros < 100)
    buffer[offset++] = '0';
  if (micros < 10)
    buffer[offset++] = '0';
  i_to_str(micros, buffer, offset);
  buffer[offset++] = 'Z';
}

void logger_t::log(logger_t::level_t level, const char* file, int line, const char* message) {
  /**
   * Writes a log entry using only async-signal-safe operations.
   *
   * This function can be safely called from signal handlers because it:
   * 1. Uses only reentrant functions
   * 2. Makes no dynamic allocations
   * 3. Modifies no global state
   * 4. Handles interrupted writes (EINTR)
   *
   * The log format is:
   * "TIMESTAMP [LEVEL] file:line: message\n"
   * Example:
   * "2024-03-27 14:30:15.123456Z [INFO] main.cpp:42: Process started\n"
   *
   * Parameters:
   *   level:   Severity level of the message
   *   file:    Source file generating the log
   *   line:    Line number in source file
   *   message: The log message to write
   *
   * Note: The entire message is assembled in a fixed buffer before
   * writing to ensure atomic log entries. The write loop handles
   * partial writes, continuing until the entire message is written
   * or an unrecoverable error occurs.
   */
  if (fd_ < 0) return;

  char buffer[256];
  int offset = 0;

  timestamp(buffer, offset);
  buffer[offset++] = ' ';
  buffer[offset++] = '[';
  for (const char* level_str = level_to_str(level); *level_str; level_str++)
    buffer[offset++] = *level_str;
  buffer[offset++] = ']';
  buffer[offset++] = ' ';
  for (const char* c = file; *c; c++)
    buffer[offset++] = *c;
  buffer[offset++] = ':';
  i_to_str(line, buffer, offset);
  buffer[offset++] = ':';
  buffer[offset++] = ' ';
  for (const char* c = message; *c; c++)
    buffer[offset++] = *c;
  buffer[offset++] = '\n';

  size_t total_bytes_written = 0;
  while (total_bytes_written < (size_t)offset) {
    ssize_t result = write(
      fd_,
      buffer + total_bytes_written,
      offset - total_bytes_written
    );

    if (result < 0) {
      if (errno == EINTR) continue;
      return;
    }

    total_bytes_written += result;
  }
}
