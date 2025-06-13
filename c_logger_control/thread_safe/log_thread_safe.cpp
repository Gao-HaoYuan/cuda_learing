#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

// ------------------------
// 用户宏配置项（可选）
// ------------------------
#ifndef LOG_TAG
#define LOG_TAG NULL
#endif

#ifndef LOG_USE_COLOR
#define LOG_USE_COLOR 1
#endif

#ifndef LOG_USE_FILE
#define LOG_USE_FILE 0
#endif

// ------------------------
// 日志等级定义
// ------------------------
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} LogLevel;

static const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        case LOG_LEVEL_FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

static const char* log_level_to_color(LogLevel level) {
    if (!LOG_USE_COLOR) return "";
    switch (level) {
        case LOG_LEVEL_DEBUG: return "\033[36m";
        case LOG_LEVEL_INFO:  return "\033[32m";
        case LOG_LEVEL_WARN:  return "\033[33m";
        case LOG_LEVEL_ERROR: return "\033[31m";
        case LOG_LEVEL_FATAL: return "\033[1;31m";
        default: return "\033[0m";
    }
}

// ------------------------
// 内部状态
// ------------------------
static LogLevel current_log_level = LOG_LEVEL_INFO;
static FILE* log_output_file = NULL;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

// ------------------------
// 初始化 & 清理
// ------------------------
static inline void log_init(LogLevel level, const char* file_path) {
    current_log_level = level;
#if LOG_USE_FILE
    if (file_path) {
        log_output_file = fopen(file_path, "a");
        if (!log_output_file) {
            perror("无法打开日志文件");
            log_output_file = stderr;
        }
    } else {
        log_output_file = stderr;
    }
#else
    (void)file_path;
    log_output_file = stderr;
#endif
}

static inline void log_close() {
#if LOG_USE_FILE
    if (log_output_file && log_output_file != stderr) {
        fclose(log_output_file);
    }
#endif
    log_output_file = NULL;
}

// ------------------------
// 日志主函数
// ------------------------
static inline void log_internal(LogLevel level, const char* tag,
                                const char* file, int line, const char* func,
                                const char* fmt, ...) {
    if (level < current_log_level) return;
    if (!log_output_file) log_output_file = stderr;

    pthread_mutex_lock(&log_mutex);

    // 时间戳
    time_t t = time(NULL);
    struct tm tm_info;
    localtime_r(&t, &tm_info);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_info);

    const char* color = log_level_to_color(level);
    const char* reset = LOG_USE_COLOR ? "\033[0m" : "";

    // 打印头部
    fprintf(log_output_file, "%s[%s] [%s] [%s] [%s:%d %s] ",
        color,
        time_buf,
        log_level_to_string(level),
        tag ? tag : "NoTag",
        file, line, func);

    // 打印正文
    va_list args;
    va_start(args, fmt);
    vfprintf(log_output_file, fmt, args);
    va_end(args);

    fprintf(log_output_file, "%s\n", reset);
    fflush(log_output_file);  // 确保立即写入

    pthread_mutex_unlock(&log_mutex);
}

#ifndef LOG_DISABLE_DEBUG
#define LOG_DEBUG(fmt, ...) log_internal(LOG_LEVEL_DEBUG, LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#endif

#define LOG_INFO(fmt, ...)  log_internal(LOG_LEVEL_INFO,  LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_internal(LOG_LEVEL_WARN,  LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_internal(LOG_LEVEL_ERROR, LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) log_internal(LOG_LEVEL_FATAL, LOG_TAG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)


////////////////////////////////////////////////////////////////////////////////
#define LOG_TAG "Main"
#define LOG_USE_COLOR 1
#define LOG_USE_FILE 1

#include "log.h"

int main() {
    log_init(LOG_LEVEL_DEBUG, "log.txt");

    LOG_INFO("服务启动成功");
    LOG_WARN("磁盘容量不足");
    LOG_DEBUG("变量值：x = %d", 42);
    LOG_ERROR("读取文件失败");
    LOG_FATAL("致命错误，终止程序");

    log_close();  // 如果你使用文件输出，记得关闭
    return 0;
}
