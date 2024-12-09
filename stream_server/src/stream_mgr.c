#include <stdio.h>

#include "logging.h"

void* stream_mgr(void* ptr) {
  char* msg = (char*) ptr;
  printf("%s\n", msg);

  log(INFO, "Logged from stream mgr");

  return NULL;
}
