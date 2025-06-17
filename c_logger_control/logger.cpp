#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

#include "logger.h"

static LogLevel current_log_level = LOG_LEVEL_DEBUG;

const char *log_level_to_color(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "\033[36m";   // 青色
        case LOG_LEVEL_INFO: return "\033[32m";    // 绿色
        case LOG_LEVEL_WARN: return "\033[33m";    // 黄色
        case LOG_LEVEL_ERROR: return "\033[31m";   // 红色
        case LOG_LEVEL_FATAL: return "\033[1;31m"; // 加粗红
        default: return "\033[0m";
    }
}

const char *log_level_to_string(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO: return "INFO";
        case LOG_LEVEL_WARN: return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        case LOG_LEVEL_FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

void log_set_level(LogLevel level) {
    current_log_level = level;
}

LogLevel get_log_level_from_env(void) {
    const char *env = getenv("LOG_LEVEL");
    if (!env) return LOG_LEVEL_INFO;

    if (strcasecmp(env, "DEBUG") == 0) return LOG_LEVEL_DEBUG;
    if (strcasecmp(env, "INFO") == 0) return LOG_LEVEL_INFO;
    if (strcasecmp(env, "WARN") == 0) return LOG_LEVEL_WARN;
    if (strcasecmp(env, "ERROR") == 0) return LOG_LEVEL_ERROR;
    if (strcasecmp(env, "FATAL") == 0) return LOG_LEVEL_FATAL;

    return LOG_LEVEL_INFO;
}

void log_internal(LogLevel level,
                  const char *tag,
                  const char *file,
                  int line,
                  const char *func,
                  const char *fmt,
                  ...) {
    if (level < current_log_level) return;

    // 时间戳
    time_t t = time(NULL);
    struct tm tm_info;
    localtime_r(&t, &tm_info);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_info);

    const char *color = log_level_to_color(level);
    const char *reset = "\033[0m";

    // 打印日志头
    fprintf(stderr,
            "%s[%s] [%s] [%s] [%s:%d %s] ",
            color,
            time_buf,
            log_level_to_string(level),
            tag ? tag : "NoTag",
            file,
            line,
            func);

    // 打印正文
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "%s\n", reset); // 恢复颜色
}

int main() {
    LOG_INFO("服务启动成功");
    LOG_WARN("磁盘容量不足");
    LOG_DEBUG("变量值: x = %d", 42);
    LOG_ERROR("读取文件失败");
    LOG_FATAL("致命错误，终止程序");

    return 0;
}