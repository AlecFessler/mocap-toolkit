#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

class config_parser {
public:
  config_parser(const std::string& filename) {
    std::ifstream file(filename);
    if (!file)
      throw std::runtime_error("Could not open config file: " + filename);

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string key, value;
      if (std::getline(iss, key, '=') && std::getline(iss, value))
        config_[key] = value;
    }
  }

  std::string get_string(const std::string& key) const {
    return get_value(key);
  }

  int get_int(const std::string& key) const {
    return std::stoi(get_value(key));
  }

private:
  std::unordered_map<std::string, std::string> config_;

  std::string get_value(const std::string& key) const {
    auto it = config_.find(key);
    if (it == config_.end())
      throw std::runtime_error("Config key not found: " + key);
    return it->second;
  }
};

#endif // CONFIG_PARSER_H
