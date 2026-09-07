// ============================================================================
//  示例: 每个进程线程管理多个 GPU 的 MCCL/MPI 组合
//
//  学习目标:
//    1. 在进程/rank 维度之外, 继续管理本地多个 device。
//    2. 理解 communicator、stream、buffer 都需要按 device 维度组织。
//    3. 对比 4_12 和 4-13, 观察控制流复杂度如何随 device/thread 映射变化。
//
//  阅读顺序:
//    先分清 MPI rank、host thread 和 local device 三个维度, 再追踪每个 device 的资源数组。
//
//  注意:
//    多 device per thread 写法灵活, 但最容易出现上下文切错、buffer 归属混乱、
//    stream 同步遗漏等问题。
// ============================================================================
#include <stdio.h>
#include <musa_runtime.h>
#include <mccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#define MPICHECK(cmd)                                                                              \
    do {                                                                                           \
        int e = cmd;                                                                               \
        if (e != MPI_SUCCESS) {                                                                    \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
#define MUSACHECK(cmd)                                                                             \
    do {                                                                                           \
        musaError_t e = cmd;                                                                       \
        if (e != musaSuccess) {                                                                    \
            printf("Failed: MUSA error %s:%d '%s'\n", __FILE__, __LINE__, musaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
#define MCCLCHECK(cmd)                                                                             \
    do {                                                                                           \
        mcclResult_t r = cmd;                                                                      \
        if (r != mcclSuccess) {                                                                    \
            printf("Failed, MCCL error %s:%d '%s'\n", __FILE__, __LINE__, mcclGetErrorString(r));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
int GetEnvWithDefault(const char *env_var, int default_val = 2) {
    const char *val = getenv(env_var);
    if (val != NULL) {
        return atoi(val);
    }
    return default_val;
}
static uint64_t GenHash(const char *string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}
// Generate a hash of the unique identifying string for this host
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t GetHostHash(const char *hostname) {
    char hosthash[1024];
    // fall back is the hostname if something fails
    (void)strncpy(hosthash, hostname, sizeof(hosthash));
    int offset = strlen(hosthash);
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
        char *p;
        if (fscanf(file, "%ms", &p) == 1) {
            strncpy(hosthash + offset, p, sizeof(hosthash) - offset - 1);
            free(p);
        }
    }
    fclose(file);
    // make sure the string is terminated
    hosthash[sizeof(hosthash) - 1] = '\0';
    return GenHash(hosthash);
}
static void GetHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}
int main(int argc, char *argv[]) {
    int size = 32 << 20;
    int my_rank, n_ranks, local_rank = 0;
    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
    // calculating local_rank based on hostname which is used in selecting a GPU
    uint64_t host_hashs[n_ranks];
    char hostname[1024];
    GetHostName(hostname, 1024);
    host_hashs[my_rank] = GetHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashs, sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < n_ranks; p++) {
        if (p == my_rank)
            break;
        if (host_hashs[p] == host_hashs[my_rank])
            local_rank++;
    }
    // using ndev GPUs per thread
    int ndev = GetEnvWithDefault("EXAMPLE_NDEV_PER_THREAD", 2);
    // allocate device ptr and streams for each device
    float **sendbuff = (float **)malloc(ndev * sizeof(float *));
    float **recvbuff = (float **)malloc(ndev * sizeof(float *));
    musaStream_t *s = (musaStream_t *)malloc(sizeof(musaStream_t) * ndev);
    // picking GPUs based on local_rank
    for (int i = 0; i < ndev; ++i) {
        MUSACHECK(musaSetDevice(local_rank * ndev + i));
        MUSACHECK(musaMalloc(sendbuff + i, size * sizeof(float)));
        MUSACHECK(musaMalloc(recvbuff + i, size * sizeof(float)));
        MUSACHECK(musaMemset(sendbuff[i], 1, size * sizeof(float)));
        MUSACHECK(musaMemset(recvbuff[i], 0, size * sizeof(float)));
        MUSACHECK(musaStreamCreate(s + i));
    }
    mcclUniqueId id;
    mcclComm_t comms[ndev];
    // generating unique ID at one process and broadcasting it to all
    if (my_rank == 0) {
        MCCLCHECK(mcclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    // initializing MCCL, group API is required around mcclCommInitRank as it is
    // called across multiple GPUs in each thread/process
    MCCLCHECK(mcclGroupStart());
    for (int i = 0; i < ndev; i++) {
        MUSACHECK(musaSetDevice(local_rank * ndev + i));
        MCCLCHECK(mcclCommInitRank(comms + i, n_ranks * ndev, id, my_rank * ndev + i));
    }
    MCCLCHECK(mcclGroupEnd());
    // calling MCCL communication API. Group API is required when using
    // multiple devices per thread/process
    MCCLCHECK(mcclGroupStart());
    for (int i = 0; i < ndev; i++) {
        MCCLCHECK(mcclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, mcclFloat,
                                mcclSum, comms[i], s[i]));
    }
    MCCLCHECK(mcclGroupEnd());
    // synchronizing on MUSA stream to complete MCCL communication
    for (int i = 0; i < ndev; i++) {
        MUSACHECK(musaStreamSynchronize(s[i]));
    }
    // freeing device memory
    for (int i = 0; i < ndev; i++) {
        MUSACHECK(musaFree(sendbuff[i]));
        MUSACHECK(musaFree(recvbuff[i]));
    }
    // finalizing MCCL
    for (int i = 0; i < ndev; i++) {
        MCCLCHECK(mcclCommDestroy(comms[i]));
    }
    // finalizing MPI
    MPICHECK(MPI_Finalize());
    printf("[MPI Rank %d][LocalRank %d][ndev %d] Success \n", my_rank, local_rank, ndev);
    return 0;
}
