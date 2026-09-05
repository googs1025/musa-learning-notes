// ============================================================================
//  示例: 每个进程线程绑定一个 GPU 的 MCCL/MPI 组合
//
//  学习目标:
//    1. 用 MPI 管理进程间 rank, 再把每个 rank 映射到一个 local device。
//    2. 每个线程只操作自己的 device, 降低频繁 musaSetDevice 的复杂度。
//    3. 通过 MCCL communicator 在多个 GPU/rank 之间执行 collective。
//
//  注意:
//    这个模型更接近分布式训练常见写法; 排错时要同时看 MPI rank、local rank
//    和 device id 的映射是否一致。
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
static uint64_t GenHash(const char *string) {
    // based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}
// Generate a hash of the unique identifying string for this host
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t GenHostHash(const char *hostname) {
    char hosthash[1024];
    // fall back is the hostname if something fails
    (void)strncpy(hosthash, hostname, sizeof(hosthash));
    int offset = strlen(hosthash);
    // try to read the host ID from the host boot ID file
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
    host_hashs[my_rank] = GenHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashs, sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < n_ranks; p++) {
        if (p == my_rank)
            break;
        if (host_hashs[p] == host_hashs[my_rank])
            local_rank++;
    }
    mcclUniqueId id;
    mcclComm_t comm;
    float *sendbuff, *recvbuff;
    musaStream_t s;
    // get MCCL unique ID at rank 0 and broadcast it to all others
    if (my_rank == 0) {
        MCCLCHECK(mcclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    // picking a GPU based on local_rank, allocate device buffers
    MUSACHECK(musaSetDevice(local_rank));
    MUSACHECK(musaMalloc(&sendbuff, size * sizeof(float)));
    MUSACHECK(musaMalloc(&recvbuff, size * sizeof(float)));
    MUSACHECK(musaMemset(sendbuff, 1, size * sizeof(float)));
    MUSACHECK(musaMemset(recvbuff, 0, size * sizeof(float)));
    MUSACHECK(musaStreamCreate(&s));
    // initializing MCCL
    MCCLCHECK(mcclCommInitRank(&comm, n_ranks, id, my_rank));
    // communicating using MCCL
    MCCLCHECK(
        mcclAllReduce((const void *)sendbuff, (void *)recvbuff, size, mcclFloat, mcclSum, comm, s));
    // completing MCCL operation by synchronizing on the MUSA stream
    MUSACHECK(musaStreamSynchronize(s));
    // free device buffers
    MUSACHECK(musaFree(sendbuff));
    MUSACHECK(musaFree(recvbuff));
    // finalizing MCCL
    MCCLCHECK(mcclCommDestroy(comm));
    // finalizing MPI
    MPICHECK(MPI_Finalize());
    printf("[MPI Rank %d][local_rank %d] Success \n", my_rank, local_rank);
    return 0;
}
