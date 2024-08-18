#include "p_ctx.h"

p_ctx_t::p_ctx_t() noexcept
  : logger(), cam(), shared_mem() {}

p_ctx_t& p_ctx_t::get_instance() noexcept {
    static p_ctx_t instance;
    return instance;
}
