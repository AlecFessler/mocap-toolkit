#ifndef PROCESS_CONTEXT_H
#define PROCESS_CONTEXT_H

#include <memory>
#include <signal.h>
#include "CameraHandler.h"
#include "Logger.h"

class CameraHandler;

class PCtx {
public:
  volatile sig_atomic_t running;
  volatile sig_atomic_t queueRequest;
  volatile sig_atomic_t requestComplete;
  std::unique_ptr<Logger> logger;
  std::unique_ptr<CameraHandler> cam;

  static PCtx& getInstance() noexcept;

  PCtx(PCtx const&) = delete;
  void operator=(PCtx const&) = delete;
  PCtx(PCtx&&) = delete;
  void operator=(PCtx&&) = delete;

private:
  PCtx() noexcept;
};

#endif // PROCESS_CONTEXT_H
