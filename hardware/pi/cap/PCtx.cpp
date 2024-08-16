#include "PCtx.h"

PCtx::PCtx() noexcept
  : running(0), queueRequest(0), requestComplete(0), logger(), cam() {}

PCtx& PCtx::getInstance() noexcept {
    static PCtx instance;
    return instance;
}
