#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <yaml.h>

#include "parse_conf.h"
#include "logging.h"

static FILE* infile = NULL;

int count_cameras(const char* fpath) {
  /**
   * Counts the instances of 'rpicam' substring appearing in the file
   *
   * If this succeeds, the file is cached and parse_conf will reuse it then clean it up
   *
   * Parameters:
   * - const char* fpath: the file path of the camera conf
   *
   * Returns:
   * - int: either a negative error code, or positive count of cameras
   */
  int ret = 0;
  char logstr[128];

  infile = fopen(fpath, "r");
  if (!infile) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error opening file: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    ret = -errno;
    goto err_cleanup;
  }

  yaml_parser_t parser;
  yaml_event_t event;

  ret = yaml_parser_initialize(&parser);
  if (ret == 0) {
    log(ERROR, "Error initializing yaml parser");
    ret = -ENOMEM;
    goto err_cleanup;
  }
  yaml_parser_set_input_file(&parser, infile);

  int count = 0;
  while (true) {
    ret = yaml_parser_parse(&parser, &event);
    if (ret == 0) {
      log(ERROR, "Error parsing yaml file");
      ret = -EINVAL;
      goto err_cleanup;
    }

    if (event.type == YAML_STREAM_END_EVENT) {
      if (count == 0) {
        log(ERROR, "Found no cameras list");
        ret = -EINVAL;
        goto err_cleanup;
      }

      ret = count;
      goto success_cleanup;
    }

    if (event.type != YAML_SCALAR_EVENT) {
      yaml_event_delete(&event);
      continue;
    }

    // count the occurences of 'rpicam'
    if (strncmp((char*)event.data.scalar.value, "rpicam", 6) == 0)
      count++;

    yaml_event_delete(&event);
  }

  err_cleanup:
  if (infile) {
    fclose(infile);
    infile = NULL;
  }

  success_cleanup:
  if (infile)
    rewind(infile);

  yaml_parser_delete(&parser);
  yaml_event_delete(&event);

  return ret;
}

typedef int (*parser_fn)(const char* str, void* field);

static int parse_str(const char* str, void* field) {
  bool inval_name = (strcmp((char*)field, "name") == 0) &&
                    (strlen(str) > CAM_NAME_LEN);
  if (inval_name)
    return -EINVAL;

  strcpy((char*)field, str);
  return 0;
}

static int parse_uint8(const char* str, void* field) {
  *(uint8_t*)field = atoi(str);
  return 0;
}

static int parse_uint16(const char* str, void* field) {
  *(uint16_t*)field = atoi(str);
  return 0;
}

static int parse_uint32(const char* str, void* field) {
  *(uint32_t*)field = atoi(str);
  return 0;
}

static int parse_ipv4(const char* str, void* field) {
  return inet_pton(AF_INET, str, field) > 0 ? 0 : -EINVAL;
}

struct field_map {
  const char* name;
  size_t offset;
  parser_fn parser;
};

static const struct field_map stream_fields[] = {
  {"frame_width", offsetof(struct stream_conf, frame_width), parse_uint32},
  {"frame_height", offsetof(struct stream_conf, frame_height), parse_uint32},
  {"fps", offsetof(struct stream_conf, fps), parse_uint32}
};

static const struct field_map fields[] = {
  {"name", offsetof(struct cam_conf, name), parse_str},
  {"id", offsetof(struct cam_conf, id), parse_uint8},
  {"eth_ip", offsetof(struct cam_conf, eth_ip), parse_ipv4},
  {"wifi_ip", offsetof(struct cam_conf, wifi_ip), parse_ipv4},
  {"tcp_port", offsetof(struct cam_conf, tcp_port), parse_uint16},
  {"udp_port", offsetof(struct cam_conf, udp_port), parse_uint16}
};

static int parse_stream_params(struct stream_conf* stream_conf) {
  int ret = 0;
  char logstr[128];

  if (!infile) {
    log(ERROR, "File not opened, call count_cameras first");
    ret = -ENODATA;
    goto cleanup;
  }

  yaml_parser_t parser;
  yaml_event_t event;

  ret = yaml_parser_initialize(&parser);
  if (ret == 0) {
    log(ERROR, "Error initializing yaml parser");
    ret = -ENOMEM;
    goto cleanup;
  }
  yaml_parser_set_input_file(&parser, infile);

  int fields_parsed = 0;
  const int fields_total = sizeof(stream_fields)/sizeof(stream_fields[0]);
  bool in_stream_params = false;

  while (true) {
    ret = yaml_parser_parse(&parser, &event);
    if (ret == 0) {
      log(ERROR, "Error parsing yaml file");
      ret = -EINVAL;
      goto cleanup;
    }

    if (event.type == YAML_STREAM_END_EVENT) {
      log(ERROR, "Reached end of file before finding all stream parameters");
      ret = -EINVAL;
      goto cleanup;
    }

    if (event.type != YAML_SCALAR_EVENT) {
      yaml_event_delete(&event);
      continue;
    }

    if (!in_stream_params) {
      if (strcmp((char*)event.data.scalar.value, "stream_params") == 0)
        in_stream_params = true;

      yaml_event_delete(&event);
      continue;
    }

    if (strcmp((char*)event.data.scalar.value, "cameras") == 0) {
      if (fields_parsed < fields_total) {
        log(ERROR, "Missing required stream parameters");
        ret = -EINVAL;
        goto cleanup;
      }

      ret = 0;
      goto cleanup;
    }

    for (int i = 0; i < fields_total; i++) {
      if (strcmp((char*)event.data.scalar.value, stream_fields[i].name) != 0)
        continue;

      ret = yaml_parser_parse(&parser, &event);
      if (ret == 0) {
        log(ERROR, "Error parsing yaml file");
        ret = -EINVAL;
        goto cleanup;
      }

      void* field = (char*)stream_conf + stream_fields[i].offset;
      ret = stream_fields[i].parser((char*)event.data.scalar.value, field);
      if (ret < 0) {
        snprintf(
          logstr,
          sizeof(logstr),
          "Failed to parse %s",
          stream_fields[i].name
        );
        log(ERROR, logstr);
        ret = -EINVAL;
        goto cleanup;
      }

      fields_parsed++;
      break;
    }

    yaml_event_delete(&event);

    if (fields_parsed == fields_total) {
      ret = 0;
      goto cleanup;
    }
  }

  cleanup:
  if (infile)
    rewind(infile);

  yaml_parser_delete(&parser);
  yaml_event_delete(&event);

  return ret;

}

int parse_conf(
  struct stream_conf* stream_conf,
  struct cam_conf* confs,
  int count
) {
  /**
   * Parses the yaml file and populates the array of structs
   *
   * Agnostic to precise ordering of fields, but only finds
   * those which are mapped in the field_map struct
   *
   * Parameters:
   * - cam_conf* confs: an array of cam_conf structs
   * - int count: the number of cam confs in the array
   *
   * Returns:
   * - int: either a negative error code, or 0 on success
   */
  int ret = 0;
  char logstr[128];

  if (!infile) {
    log(ERROR, "File not opened, call count_cameras first");
    ret = -ENODATA;
    goto cleanup;
  }

  ret = parse_stream_params(stream_conf);
  if (ret != 0)
    goto cleanup;

  yaml_parser_t parser;
  yaml_event_t event;

  ret = yaml_parser_initialize(&parser);
  if (ret == 0) {
    log(ERROR, "Error initializing yaml parser");
    ret = -ENOMEM;
    goto cleanup;
  }
  yaml_parser_set_input_file(&parser, infile);

  int confs_parsed = 0;
  int fields_parsed = 0;
  const int fields_total = sizeof(fields)/sizeof(fields[0]);

  while (confs_parsed < count) {
    ret = yaml_parser_parse(&parser, &event);
    if (ret == 0) {
      log(ERROR, "Error parsing yaml file");
      ret = -EINVAL;
      goto cleanup;
    }

    if (event.type == YAML_STREAM_END_EVENT) {
      snprintf(
        logstr,
        sizeof(logstr),
        "File ended before parsing expected conf count: %u, parsed: %u",
        count,
        confs_parsed
      );
      log(ERROR, logstr);
      ret = -EINVAL;
      goto cleanup;
    }

    if (event.type != YAML_SCALAR_EVENT) {
      yaml_event_delete(&event);
      continue;
    }

    for (int i = 0; i < fields_total; i++) {
      // check if we have a key
      ret = strcmp((char*)event.data.scalar.value, fields[i].name);
      if (ret != 0)
        continue;

      ret = yaml_parser_parse(&parser, &event);
      if (ret == 0) {
        log(ERROR, "Error parsing yaml file");
        ret = -EINVAL;
        goto cleanup;
      }

      void* field = (char*)&confs[confs_parsed] + fields[i].offset;

      ret = fields[i].parser((char*)event.data.scalar.value, field);
      if (ret < 0) {
        snprintf(
          logstr,
          sizeof(logstr),
          "Failed to parse %s",
          fields[i].name
        );
        log(ERROR, logstr);
        ret = -EINVAL;
        goto cleanup;
      }

      if (++fields_parsed == fields_total) {
        fields_parsed = 0;
        confs_parsed++;
      }
    }

    yaml_event_delete(&event);
  }

  cleanup:
  if (infile)
    fclose(infile);

  yaml_parser_delete(&parser);
  yaml_event_delete(&event);

  return ret;
}
