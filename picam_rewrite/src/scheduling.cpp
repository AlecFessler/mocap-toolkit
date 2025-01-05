#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sched.h>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "logging.hpp"
#include "scheduling.hpp"

void pin_to_core(uint32_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  int status = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (status < 0) {
    std::string err_msg =
      "Failed to pin thread to core: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}

void set_scheduling_prio(uint32_t prio) {
  struct sched_param param;
  param.sched_priority = prio;
  int status = sched_setscheduler(0, SCHED_FIFO, &param);
  if (status < 0) {
    std::string err_msg =
      "Failed to set real time scheduling priority: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}
