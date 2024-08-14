#ifndef VFQ_H
#define VFQ_H

#include <sys/ipc.h>
#include <sys/msg.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

constexpr key_t KEY = 0x1234;
constexpr int FRAME_WIDTH = 1280;
constexpr int FRAME_HEIGHT = 720;
constexpr int COLOR_CHANNELS = 3;
constexpr size_t FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * COLOR_CHANNELS;
constexpr long MSG_TYPE = 1;

typedef struct {
  long mtype;
  unsigned char frame[FRAME_SIZE];
} VideoFrame;

int create_vfq(key_t key) {
  int msg_qid = msgget(key, IPC_CREAT | 0666);
  if (msg_qid == -1) {
    perror("Failed to create message queue");
    exit(1);
  }
  return msg_qid;
}

void write_frame(int msg_qid, const unsigned char* frame) {
  VideoFrame vf;
  vf.mtype = MSG_TYPE;
  memcpy(vf.frame, frame, FRAME_SIZE);

  // IPC_NOWAIT: return immediately if the message queue is full
  if (msgsnd(msg_qid, &vf, sizeof(vf.frame), IPC_NOWAIT) == -1) {
    perror("Failed to send message");
    exit(1);
  }
}

void read_frame(int msg_qid, unsigned char* frame) {
  VideoFrame vf;

  if (msgrcv(msg_qid, &vf, sizeof(vf.frame), MSG_TYPE, 0) == -1) {
    perror("Failed to receive message");
    exit(1);
  }

  memcpy(frame, vf.frame, FRAME_SIZE);
}

void destroy_vfq(int msg_qid) {
  if (msgctl(msg_qid, IPC_RMID, NULL) == -1) {
    perror("Failed to destroy message queue");
    exit(1);
  }
}

#endif // VFQ_H
