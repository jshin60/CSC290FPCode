

#pragma once

#include "common.h"

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&m_start);
        //m_start);
        cudaEventCreate(&m_end);
        //HGEMM_CHECK(m_end);
    }

    ~CudaTimer() {
        if (m_start) {
            cudaEventDestroy(m_start);
            m_start = nullptr;
        }

        if (m_end) {
            cudaEventDestroy(m_end);
            m_end = nullptr;
        }
    }

    void start() {
        cudaEventRecord(m_start);
    }

    float end() {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        cudaEventElapsedTime(&m_elapsed_time, m_start, m_end);

        return m_elapsed_time;
    }

private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};
