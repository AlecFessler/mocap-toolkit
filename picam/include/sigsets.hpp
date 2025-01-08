// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef SIGSETS_HPP
#define SIGSETS_HPP

#include <csignal>
#include <initializer_list>

typedef void (*signal_handler_t)(int);

sigset_t setup_sigwait(int signum);
sigset_t setup_sigwait(std::initializer_list<int> signum);
void setup_sig_handler(int signum, signal_handler_t handler);

#endif // SIGSETS_HPP
