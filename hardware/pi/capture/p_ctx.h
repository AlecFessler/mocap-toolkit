#ifndef PROCESS_CONTEXT_H
#define PROCESS_CONTEXT_H

#include <atomic>
#include <memory>
#include <signal.h>
#include "camera_handler.h"
#include "shared_defs.h"
#include "logger.h"

class camera_handler_t;

class p_ctx_t {
public:
  std::unique_ptr<logger_t> logger;
  std::unique_ptr<camera_handler_t> cam;
  shared_mem_layout* shared_mem;

  static p_ctx_t& get_instance() noexcept;

  p_ctx_t(p_ctx_t const&) = delete;
  void operator=(p_ctx_t const&) = delete;
  p_ctx_t(p_ctx_t&&) = delete;
  void operator=(p_ctx_t&&) = delete;

private:
  p_ctx_t() noexcept;
};

#endif // PROCESS_CONTEXT_H
