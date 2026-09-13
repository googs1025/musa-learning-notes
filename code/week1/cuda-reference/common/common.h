/*
 * Minimal compatibility subset for the upstream common.h.
 *
 * Source: kriegalex/wrox-pro-cuda-c, MIT-licensed common utilities.
 * The full upstream header is intentionally not copied before its consumers
 * are imported. Keep this header limited to the CUDA/MUSA runtime compatibility
 * layer; it does not add other MUSA-specific dependencies.
 */

#ifndef CUDA_REFERENCE_COMMON_H
#define CUDA_REFERENCE_COMMON_H

#include <sys/time.h>

#ifdef CUDA_REFERENCE_MUSA
#include <musa_runtime.h>
#define CUDA_REFERENCE_ERROR_T musaError_t
#define CUDA_REFERENCE_SUCCESS musaSuccess
#define CUDA_REFERENCE_GET_ERROR_STRING musaGetErrorString
#else
#include <cuda_runtime.h>
#define CUDA_REFERENCE_ERROR_T cudaError_t
#define CUDA_REFERENCE_SUCCESS cudaSuccess
#define CUDA_REFERENCE_GET_ERROR_STRING cudaGetErrorString
#endif

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        CUDA_REFERENCE_ERROR_T error__ = (call);                            \
        if (error__ != CUDA_REFERENCE_SUCCESS) {                            \
            fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                    CUDA_REFERENCE_GET_ERROR_STRING(error__));              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Upstream Professional CUDA C samples use CHECK(...). */
#define CHECK(call) CUDA_CHECK(call)

/* Compatible with the timer samples in the upstream common/common.h. */
static inline double seconds(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6;
}

#endif
