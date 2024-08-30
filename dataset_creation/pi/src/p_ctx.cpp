#include "p_ctx.h"

p_ctx_t::p_ctx_t() noexcept
  :
  logger(),
  cam(),
  available_buffers(),
  frame_queue(),
  running(false),
  thread1_ready(),
  thread2_ready(),
  queue_counter() {}

p_ctx_t& p_ctx_t::get_instance() noexcept {
    static p_ctx_t instance;
    return instance;
}
