#include "PCtx.h"

PCtx::PCtx() noexcept
  : running(0), logger(), cam() {}

PCtx& PCtx::getInstance() noexcept {
    static PCtx instance;
    return instance;
}
