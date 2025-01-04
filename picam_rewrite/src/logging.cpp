// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>

#include "logging.hpp"

static int fd = -1;

Logging::Logging(const char* fpath) {
  /**
   * Provides RAII cleanup for the file descriptor
   */
  fd = open(
    fpath,
    O_WRONLY | O_CREAT | O_APPEND,
    0664
  );
  if (fd < 0) {
    std::string err_msg =
      "Failed to open log file: "
      + std::string(fpath)
      + " error: "
      + std::string(strerror(errno));
    throw std::runtime_error(err_msg);
  }
}

void Logging::setup_logging(const char* fpath) {
  static Logging instance{fpath};
}

Logging::~Logging() {
  /**
   * Provides RAII cleanup for the file descriptor
   */
  if (fd >= 0)
    close(fd);
}

static void i_to_str(
  int value,
  char* buffer,
  size_t* offset
) {
  /**
   * Converts integers to strings
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
   * Note: Caller must ensure buffer has sufficient space for maximum
   * possible number of digits plus sign (11 chars for 32-bit int)
   */
  if (value == 0) {
    buffer[(*offset)++] = '0';
    return;
  }

  if (value < 0) {
    buffer[(*offset)++] = '-';
    value = -value;
  }

  char temp[16];
  int len = 0;
  while (value > 0) {
    temp[len++] = '0' + value % 10;
    value /= 10;
  }

  while (len > 0) {
    buffer[(*offset)++] = temp[--len];
  }
}

#define LEAP_YEAR(year) ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)
#define SECONDS_PER_LEAP_YEAR 31622400
#define SECONDS_PER_YEAR 31536000
#define SECONDS_PER_DAY 86400
#define SECONDS_PER_HOUR 3600
#define SECONDS_PER_MINUTE 60
#define NANOS_PER_MILLISECOND 1000000
#define NANOS_PER_MICROSECOND 1000

static const int SECONDS_PER_MONTH[] = {
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

static void timestamp(char* buffer, size_t* offset) {
  /**
   * Generates an ISO 8601 timestamp without using unsafe time functions.
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
   * Note: The Z suffix indicates UTC timezone, which is what
   * CLOCK_REALTIME provides on Linux systems
   */
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  time_t seconds = ts.tv_sec;
  int nanos = ts.tv_nsec;

  int year = 1970;
  while (true) {
    int seconds_in_year =
      LEAP_YEAR(year) ?
      SECONDS_PER_LEAP_YEAR :
      SECONDS_PER_YEAR;

    if (seconds < seconds_in_year)
      break;

    seconds -= seconds_in_year;
    year++;
  }

  int month = 1;
  while (true) {
    int seconds_in_month =
      LEAP_YEAR(year) && month == 2 ?
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
  buffer[(*offset)++] = '-';

  if (month < 10)
    buffer[(*offset)++] = '0';

  i_to_str(month, buffer, offset);
  buffer[(*offset)++] = '-';

  if (day < 10)
    buffer[(*offset)++] = '0';

  i_to_str(day, buffer, offset);
  buffer[(*offset)++] = ' ';

  if (hour < 10)
    buffer[(*offset)++] = '0';

  i_to_str(hour, buffer, offset);
  buffer[(*offset)++] = ':';

  if (minute < 10)
    buffer[(*offset)++] = '0';

  i_to_str(minute, buffer, offset);
  buffer[(*offset)++] = ':';

  if (seconds < 10)
    buffer[(*offset)++] = '0';

  i_to_str(seconds, buffer, offset);
  buffer[(*offset)++] = '.';

  if (millis < 100)
    buffer[(*offset)++] = '0';

  if (millis < 10)
    buffer[(*offset)++] = '0';

  i_to_str(millis, buffer, offset);

  if (micros < 100)
    buffer[(*offset)++] = '0';

  if (micros < 10)
    buffer[(*offset)++] = '0';

  i_to_str(micros, buffer, offset);

  buffer[(*offset)++] = 'Z';
}

static const char* log_levels[] = {
  "[INFO]",
  "[DEBUG]",
  "[WARNING]",
  "[ERROR]",
  "[UNKNOWN]"
};

void log_write(log_level lvl, const char* file, int line, const char* log_str) {
  /**
   * Atomic writes a log entry, safe for multiple threads to share
   * Also safe for usage within signal handlers
   *
   * The log format is:
   * "TIMESTAMP [LEVEL] file:line: message\n"
   * Example:
   * "2024-03-27 14:30:15.123456Z [INFO] main.cpp:42: Process started\n"
   */
  if (fd < 0)
    return;

  char buffer[256];
  size_t offset = 0;

  timestamp(buffer, &offset);
  buffer[offset++] = ' ';
  const char* level_str = log_levels[lvl];
  for (; *level_str; level_str++)
    buffer[offset++] = *level_str;

  buffer[offset++] = ' ';
  for (const char* c = file; *c; c++)
    buffer[offset++] = *c;

  buffer[offset++] = ':';
  i_to_str(line, buffer, &offset);
  buffer[offset++] = ':';
  buffer[offset++] = ' ';
  for (const char* c = log_str; *c; c++)
    buffer[offset++] = *c;

  buffer[offset++] = '\n';

  size_t total_bytes_written = 0;
  while (total_bytes_written < offset) {
    ssize_t result = write(
      fd,
      buffer + total_bytes_written,
      offset - total_bytes_written
    );

    if (result < 0) {
      if (errno == EINTR)
        continue;

      return;
    }

    total_bytes_written += result;
  }
}
