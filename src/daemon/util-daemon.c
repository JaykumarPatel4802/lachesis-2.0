#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sqlite3.h>

#define MAX_INITIAL_CONTAINERS 4
char existingContainers[MAX_INITIAL_CONTAINERS][128]; // Global list to store existing containers
int existingContainersCount = 0; // Count of existing containers

// Function to read the names of existing containers
void obtainInitContainer() {
    DIR *containerDir = opendir("/sys/fs/cgroup/memory/docker");

    if (containerDir != NULL) {
        struct dirent *entry;
        while ((entry = readdir(containerDir)) != NULL) {
            if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                if (existingContainersCount < MAX_INITIAL_CONTAINERS) {
                    strcpy(existingContainers[existingContainersCount], entry->d_name);
                    existingContainersCount++;
                }
            }
        }
        closedir(containerDir);
    }
}

// Function to check if a container name is in the existing containers list
int isInitContainer(const char *containerName) {
    for (int i = 0; i < existingContainersCount; i++) {
        if (strcmp(existingContainers[i], containerName) == 0) {
            return 1; // Container is in the existing containers list
        }
    }
    return 0; // Container is not in the existing containers list
}

// Function to read an integer from a file
long double readValueFromFile(const char *filename) {
    long double value = 0;
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fscanf(file, "%Lf", &value);
    fclose(file);
    return value;
}

// Function to process a container
void processContainer(const char *containerName, sqlite3 *db) {

    // 1. Check if the container is one of the initial containers before running any invocation
    if (isInitContainer(containerName)) {
        return;
    }

    // 2. Get memory utilization
    char memory_path[1024];
    sprintf(memory_path, "/sys/fs/cgroup/memory/docker/%s", containerName);

    char memory_limit_path[2048];
    char memory_usage_path[2048];
    char memory_inactive_path[2048];
    sprintf(memory_limit_path, "%s/memory.limit_in_bytes", memory_path);
    sprintf(memory_usage_path, "%s/memory.usage_in_bytes", memory_path);
    sprintf(memory_inactive_path, "%s/memory.stat", memory_path);

    long double memory_limit = readValueFromFile(memory_limit_path) / (1024 * 1024);
    long double memory_usage = readValueFromFile(memory_usage_path);
    int total_inactive_file = 0;

    FILE *memory_inactive_file = fopen(memory_inactive_path, "r");
    if (memory_inactive_file != NULL) {
        char line[128];
        while (fgets(line, sizeof(line), memory_inactive_file)) {
            if (strstr(line, "total_inactive_file")) {
                sscanf(line, "total_inactive_file %d", &total_inactive_file);
                break;
            }
        }
        fclose(memory_inactive_file);
    }

    int memory_util = (memory_usage > total_inactive_file) ? ((memory_usage - total_inactive_file) / (1024 * 1024)) : 0;

    // 3. Get CPU utilization
    char cpu_path[1024];
    sprintf(cpu_path, "/sys/fs/cgroup/cpu/docker/%s", containerName);

    char cpu_usage_path[2048];
    char cpu_percpu_path[2048];
    sprintf(cpu_usage_path, "%s/cpuacct.usage", cpu_path);
    sprintf(cpu_percpu_path, "%s/cpuacct.usage_percpu", cpu_path);


    // Read and calculate CPU metrics
    long double cpu_usage_ns = readValueFromFile(cpu_usage_path) / 1e9;
    FILE *usage_percpu_file = fopen(cpu_percpu_path, "r");
    int num_cores = 0;
    if (usage_percpu_file != NULL) {
        char line[128];
        while (fgets(line, sizeof(line), usage_percpu_file)) {
            char *token = strtok(line, " ");
            while (token != NULL) {
                long double usage = strtold(token, NULL);
                num_cores++; // Count all values, regardless of their magnitude
                token = strtok(NULL, " ");
            }
        }
        fclose(usage_percpu_file);
    }

    FILE *proc_stat_file = fopen("/proc/stat", "r");
    long double curr_system_usage = 0.0;
    if (proc_stat_file != NULL) {
        char line[128];
        if (fgets(line, sizeof(line), proc_stat_file)) {
            long long user, nice, system, idle, iowait, irq, softirq;
            sscanf(line, "cpu %lld %lld %lld %lld %lld %lld %lld", &user, &nice, &system, &idle, &iowait, &irq, &softirq);
            curr_system_usage = (user + nice + system + idle + iowait + irq + softirq) / 100.0;
        }
        fclose(proc_stat_file);
    }
    
    // 4. Get network utilization
    // char container_pid_path[1024];
    // sprintf(container_pid_path, "/sys/fs/cgroup/memory/docker/%s/cgroup.procs", containerName);
    // FILE *container_pid_file = fopen(container_pid_path, "r");
    // if (container_pid_file == NULL) {
    //     perror("Error opening container PID file");
    //     return;
    // }

    // pid_t container_pid;
    // fscanf(container_pid_file, "%d", &container_pid);
    // fclose(container_pid_file);

    // char net_dev_path[1024];
    // sprintf(net_dev_path, "/proc/%d/net/dev", container_pid);

    // FILE *net_dev_file = fopen(net_dev_path, "r");
    // if (net_dev_file == NULL) {
    //     perror("Error opening net/dev file");
    //     return;
    // }

    // char line[2048];
    // // Skip the first two lines (header lines) in net/dev
    // fgets(line, sizeof(line), net_dev_file);
    // fgets(line, sizeof(line), net_dev_file);

    // // Read the eth0 line and get the bytes received and transmitted
    // fgets(line, sizeof(line), net_dev_file);
    // long long bytes_received, bytes_transmitted;
    // sscanf(line, "%*s %lld %*s %*s %*s %*s %*s %*s %*s %lld", &bytes_received, &bytes_transmitted);
    // fclose(net_dev_file);

    // printf("Bytes received %lld: \n", bytes_received);
    // printf("Bytes transmitted %lld: \n\n", bytes_transmitted);

    // Insert data into the SQLite table
    char insertSQL[1024];
    sprintf(insertSQL, "INSERT INTO function_utilization_advanced (container_id, timestamp, cpu_usage_ns, num_cores, curr_system_usage, mem_util, mem_limit) VALUES ('%s', strftime('%%Y-%%m-%%dT%%H:%%M:%%fZ', 'now'), %.6Lf, %d, %6Lf, %d, %.6Lf);", containerName, cpu_usage_ns, num_cores, curr_system_usage, memory_util, memory_limit);

    int rc = sqlite3_exec(db, insertSQL, 0, 0, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
    }
}

int main() {

    sqlite3 *db;
    char *errMessage = 0;
    int rc;

    // Open the database and set WAL mode
    rc = sqlite3_open("invoker_data.db", &db);
    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return (1);
    }
    
    // Set WAL mode
    rc = sqlite3_exec(db, "PRAGMA journal_mode=WAL;", 0, 0, &errMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "PRAGMA error: %s\n", errMessage);
        sqlite3_free(errMessage);
    }

    // Create the table if it doesn't exist
    char *createTableSQL = "CREATE TABLE IF NOT EXISTS function_utilization_advanced (container_id TEXT, timestamp TEXT, cpu_usage_ns REAL, num_cores REAL, curr_system_usage REAL, mem_util REAL, mem_limit REAL);";
    rc = sqlite3_exec(db, createTableSQL, 0, 0, &errMessage);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errMessage);
        sqlite3_free(errMessage);
    }
    
    // Read the existing containers at the beginning - don't need to monitor these
    obtainInitContainer();

    while (1) {
        // Open the directory for container names
        DIR *containerDir = opendir("/sys/fs/cgroup/memory/docker");

        if (containerDir != NULL) {
            struct dirent *entry;
            while ((entry = readdir(containerDir)) != NULL) {
                if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                    char containerName[128];
                    strcpy(containerName, entry->d_name);
                    processContainer(containerName, db);
                }
            }
            closedir(containerDir);
        }

        usleep(500000); // 500 milliseconds in microseconds
    }
    sqlite3_close(db);
    return 0;
}