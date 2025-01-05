#ifndef SCHEDULING_HPP
#define SCHEDULING_HPP

#include <cstdint>

void pin_to_core(uint32_t core);
void set_scheduling_prio(uint32_t prio);

#endif // SCHEDULING_HPP
