#pragma once

typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} LogLevel;

void log_set_level(LogLevel level);
LogLevel get_log_level_from_env(void);
const char* log_level_to_color(LogLevel level);
const char* log_level_to_string(LogLevel level);
void log_internal(LogLevel level, const char* tag,
                                const char* file, int line,
                                const char* func,
                                const char* fmt, ...);

#ifndef LOG_TAG
#define LOG_TAG NULL
#endif

// 编译期可关闭 DEBUG 日志
#ifndef LOG_DISABLE_DEBUG
#define LOG_DEBUG(fmt, ...) log_internal(LOG_LEVEL_DEBUG, LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#endif

#define LOG_INFO(fmt, ...)  log_internal(LOG_LEVEL_INFO,  LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_internal(LOG_LEVEL_WARN,  LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_internal(LOG_LEVEL_ERROR, LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) log_internal(LOG_LEVEL_FATAL, LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
