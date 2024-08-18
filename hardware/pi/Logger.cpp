#include "Logger.h"

#include <cstring>
#include <fcntl.h>
#include <sys/time.h>
#include <stdexcept>
#include <time.h>
#include <unistd.h>

Logger::Logger(const char* filename) {
  fd_ = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd_ < 0)
    throw std::runtime_error("Failed to open log file");
}

Logger::~Logger() {
  if (fd_ >= 0) close(fd_);
}

static const char* levelToStr(Logger::Level level) {
  constexpr const char* levels[] = {"INFO", "WARNING", "ERROR", "UNKNOWN"};
  switch (level) {
    case Logger::Level::INFO: return levels[0];
    case Logger::Level::WARNING: return levels[1];
    case Logger::Level::ERROR: return levels[2];
    default: return levels[3];
  }
}

static void intToStr(int value, char* buffer, int& offset) {
  /**
   * Async-signal-safe integer to string conversion
   *
   * This function converts an integer to a string and stores it in the buffer
   * Caller is responsible for ensuring the buffer is large enough to store the string
   *
   * args:
   *  value: the integer to convert
   *  buffer: the buffer to store the string
   *  offset: the offset in the buffer to store the string
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
   * Async-signal-safe timestamp generator
   *
   * This function generates a timestamp in the format "YYYY-MM-DD HH:MM:SS.mmmuuuZ"
   * Requires a buffer of at least 28 bytes to store the timestamp, to leave room
   * for the null terminator, though this function itself does not null terminate the string.
   *
   * args:
   *  buffer: a buffer of 28 bytes to store the timestamp
   */
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  time_t seconds = ts.tv_sec;
  int nanos = ts.tv_nsec;

  int year = 1970;
  while (true) {
    int secondsInYear = LEAP_YEAR(year) ?
                        SECONDS_PER_LEAP_YEAR :
                        SECONDS_PER_YEAR;
    if (seconds < secondsInYear)
      break;
    seconds -= secondsInYear;
    year++;
  }

  int month = 1;
  while (true) {
    int secondsInMonth = LEAP_YEAR(year) && month == 2 ?
                         SECONDS_PER_MONTH[0] :
                         SECONDS_PER_MONTH[month];
    if (seconds < secondsInMonth)
      break;
    seconds -= secondsInMonth;
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

  intToStr(year, buffer, offset);
  buffer[offset++] = '-';
  if (month < 10)
    buffer[offset++] = '0';
  intToStr(month, buffer, offset);
  buffer[offset++] = '-';
  if (day < 10)
    buffer[offset++] = '0';
  intToStr(day, buffer, offset);
  buffer[offset++] = ' ';
  if (hour < 10)
    buffer[offset++] = '0';
  intToStr(hour, buffer, offset);
  buffer[offset++] = ':';
  if (minute < 10)
    buffer[offset++] = '0';
  intToStr(minute, buffer, offset);
  buffer[offset++] = ':';
  if (seconds < 10)
    buffer[offset++] = '0';
  intToStr(seconds, buffer, offset);
  buffer[offset++] = '.';
  if (millis < 100)
    buffer[offset++] = '0';
  if (millis < 10)
    buffer[offset++] = '0';
  intToStr(millis, buffer, offset);
  if (micros < 100)
    buffer[offset++] = '0';
  if (micros < 10)
    buffer[offset++] = '0';
  intToStr(micros, buffer, offset);
  buffer[offset++] = 'Z';
}

void Logger::log(Logger::Level level, const char* file, int line, const char* message) {
  /**
   * Async-signal-safe logging function
   *
   * This function logs a message to the log file in the format
   * "YYYY-MM-DD HH:MM:SS.mmmuuuZ [LEVEL] file:line: message\n"
   *
   * args:
   *  level: the log level
   *  file: the file name where the log message originated
   *  line: the line number where the log message originated
   *  message: the log message
   */
  if (fd_ < 0) return;

  char buffer[256];
  int offset = 0;

  timestamp(buffer, offset);
  buffer[offset++] = ' ';
  buffer[offset++] = '[';
  for (const char* levelStr = levelToStr(level); *levelStr; levelStr++)
    buffer[offset++] = *levelStr;
  buffer[offset++] = ']';
  buffer[offset++] = ' ';
  for (const char* c = file; *c; c++)
    buffer[offset++] = *c;
  buffer[offset++] = ':';
  intToStr(line, buffer, offset);
  buffer[offset++] = ':';
  buffer[offset++] = ' ';
  for (const char* c = message; *c; c++)
    buffer[offset++] = *c;
  buffer[offset++] = '\n';

  write(fd_, buffer, offset);
}
