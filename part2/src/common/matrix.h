#pragma once

#include <random>
#include <tuple>
#include <vector>
#include <iostream>
#include <string>
#include <fstream> 
#include <immintrin.h>

#include "common.h"
#include "mmio_highlevel.h"

#define MMA_M 16
#define MMA_N 1
#define MMA_K 16


class Matrix {
public:
    Matrix(size_t row, size_t col, const std::string &name = "Matrix", float min = -1.0, float max = 1.0)
        : m_row(row), m_col(col), m_name(name), m_min(min), m_max(max) {
        //HGEMM_CHECK_GT(m_row, 0);
        //HGEMM_CHECK_GT(m_col, 0);

        m_elem_num = m_row * m_col;
        //HGEMM_CHECK_GT(m_elem_num, 0);

        m_host_ptr = new half[m_elem_num];
        //HGEMM_CHECK(m_host_ptr);
        cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half));
        //HGEMM_CHECK(m_dev_ptr);

        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(m_min, m_max);
        for (size_t i = 0; i < m_elem_num; ++i) {
            //m_host_ptr[i] = __float2half(uniform(engine));
            m_host_ptr[i] = __float2half(1);
        }

        cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice);

        HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col, m_host_ptr, m_dev_ptr);
    }

    ~Matrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            cudaFree((void *)m_dev_ptr);
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() const {
        return m_row;
    }

    size_t getCol() const {
        return m_col;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    half *getHostPtr() const {
        return m_host_ptr;
    }

    half *getDevPtr() const {
        return m_dev_ptr;
    }

    void tearUp(Matrix *base) {
        //HGEMM_CHECK(base);
        //HGEMM_CHECK_EQ(m_row, base->getRow());
        //HGEMM_CHECK_EQ(m_col, base->getCol());

        
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(half), cudaMemcpyDeviceToDevice);
    }

    void moveToHost() {
        cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(half), cudaMemcpyDeviceToHost);
    }

    void moveToDevice() {
        cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice);
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(half));
    }

    void memSetDevice() {
        cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half));
    }

    void checkValue(Matrix *base) {
        //HGEMM_CHECK(base);
        //HGEMM_CHECK_EQ(m_row, base->getRow());
        //HGEMM_CHECK_EQ(m_col, base->getCol());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) - __half2float(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
            //HLOG("%f", diff);
        }
        //HLOG("%d", m_elem_num);
        m_avg_diff /= static_cast<double>(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

private:
    const size_t m_row = 0;
    const size_t m_col = 0;
    const std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the hgemm results
    const float m_min = -1.0;
    const float m_max = 1.0;

    size_t m_elem_num = 0;
    half *m_host_ptr = nullptr;
    half *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(Matrix);
};



class SparseMatrix {
public:
    SparseMatrix(const std::string &name = "Matrix", char* file = "./data/cop20k_A.mtx")
        : m_name(name), filename(file) {

        readCsr();

        HLOG("Read %s", filename);
        //outputCsr();
        //HGEMM_CHECK_GT(m_row, 0);
        //HGEMM_CHECK_GT(m_col, 0);
        //HGEMM_CHECK_GT(nnz, 0);

        HLOG("%zu x %zu, nnz = %zu, A[0] = %f", m_row, m_col, nnz, __half2float(csrVal_host[0]));

        cudaMalloc((void **)&csrVal_dev, nnz * sizeof(half));
        cudaMalloc((void **)&csrColIdx_dev, nnz * sizeof(int));
        cudaMalloc((void **)&csrRowPtr_dev, (m_row + 1) * sizeof(int));

        //HGEMM_CHECK(csrVal_dev);
        //HGEMM_CHECK(csrColIdx_dev);
        //HGEMM_CHECK(csrRowPtr_dev);

        //HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));

        cudaMemcpy(csrVal_dev, csrVal_host, nnz * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIdx_dev, csrColIdx_host, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrRowPtr_dev, csrRowPtr_host, (m_row + 1) * sizeof(int), cudaMemcpyHostToDevice);


        csrToBcsr();
        //csrToBcsrKnapsacking();
        HLOG("Finished creating BCSR from CSR");
        HLOG("%zu total blocks, %zu nonzero blocks, %zu dense blocks, %zu sparse blocks", numberOfBlocks, nonzeroBlocks, denseBlocks, sparseBlocks);
        // HLOG("%d block is nonzero", blockInfo[0]);
        /* for (int i = 0; i < MMA_M; i++) {
            for (int j = 0; j < MMA_K; j++) {
                printf("%4.2f ", __half2float(bcsrVal_host[i * MMA_K + j]));
            }
            printf("\n");
        } */

        cudaMalloc((void **)&bcsrVal_dev, nonzeroBlocks * MMA_M * MMA_K * sizeof(half));
        cudaMalloc((void **)&bcsrColIdx_dev, nonzeroBlocks * sizeof(int));
        cudaMalloc((void **)&bcsrRowPtr_dev, (m_row / MMA_M + 1) * sizeof(int));
        cudaMalloc((void **)&blockInfo_dev, numberOfBlocks * sizeof(int));
        cudaMalloc((void **)&relativeBlockIndexMapping_dev, numberOfBlocks * sizeof(int));

        //HGEMM_CHECK(bcsrVal_dev);
        //HGEMM_CHECK(bcsrColIdx_dev);
        //HGEMM_CHECK(bcsrRowPtr_dev);

        cudaMemcpy(bcsrVal_dev, bcsrVal_host, nonzeroBlocks * MMA_M * MMA_K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(bcsrColIdx_dev, bcsrColIdx_host, nonzeroBlocks * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(bcsrRowPtr_dev, bcsrRowPtr_host, (m_row / MMA_M + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(blockInfo_dev, blockInfo_host, numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(relativeBlockIndexMapping_dev, relativeBlockIndexMapping_host, numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice);

        
    }

    ~SparseMatrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            cudaFree((void *)m_dev_ptr);
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() {
        return m_row;
    }

    size_t getCol() {
        return m_col;
    }

    size_t getElemNum() {
        return m_elem_num;
    }

    size_t getNnz() {
        return nnz;
    }

    half *getHostPtr() {
        return m_host_ptr;
    }

    half *getDevPtr() {
        return m_dev_ptr;
    }

    half *getBcsrValues() {
        return bcsrVal_dev;
    }

    int *getBcsrRowPtr() {
        return bcsrRowPtr_dev;
    }

    int *getBcsrColIdx() {
        return bcsrColIdx_dev;
    }

    int *getRelativeBlockIndexMapping_dev() {
        return relativeBlockIndexMapping_dev;
    }

    size_t getNonzeroblocks() {
        return nonzeroBlocks;
    }

    int *getBlockInfo_dev() {
        return blockInfo_dev;
    }

    int *getBlockInfo_host() {
        return blockInfo_host;
    }


    void tearUp(Matrix *base) {
        //HGEMM_CHECK(base);
        //HGEMM_CHECK_EQ(m_row, base->getRow());
        //HGEMM_CHECK_EQ(m_col, base->getCol());

        
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(half), cudaMemcpyDeviceToDevice);
    }

    void moveToHost() {
        cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(half), cudaMemcpyDeviceToHost);
    }

    void moveToDevice() {
        cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice);
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(half));
    }

    void memSetDevice() {
        cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half));
    }

    void checkValue(Matrix *base) {
        //HGEMM_CHECK(base);
        //HGEMM_CHECK_EQ(m_row, base->getRow());
        //HGEMM_CHECK_EQ(m_col, base->getCol());
        if (m_elem_num == 0) {
            m_elem_num = m_row * m_col;
        }
        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) - __half2float(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
        }

        m_avg_diff /= static_cast<double>(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

    void readCsr() {
        int isSymmetric;
        mmio_allinone(&m_row, &m_col, &nnz, &isSymmetric, &csrRowPtr_host, &csrColIdx_host, &csrVal_host, filename);

    }

    void outputCsr() {
        // Open file_transformed.mtx for writing
        std::string file_name_str(filename);
        std::ofstream outfile("./data/magicube_cop20k_A.mtx");

        // Check if file is opened successfully
        if (!outfile) {
            std::cerr << "Failed to open file_transformed.mtx for writing." << std::endl;
            return; // Exit with error
        }

        // Write the values of a, b, c separated by commas in the first line
        outfile << m_row << ", " << m_col << ", " << nnz << std::endl;

        // Write the elements of array t separated by spaces in the second line
        for (int i = 0; i < m_row + 1; ++i) {
            outfile << csrRowPtr_host[i] << " ";
        }
        outfile << std::endl;

        for (int i = 0; i < nnz; ++i) {
            outfile << csrColIdx_host[i] << " ";
        }
        outfile << std::endl;
        // Close the file
        outfile.close();
        HLOG("Outputed to Magicube format.");
    }

    void makeDenseArray() {
        m_elem_num = m_row * m_col;
        m_host_ptr = new half[m_elem_num];
        //HGEMM_CHECK(m_host_ptr);
        cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half));
        //HGEMM_CHECK(m_dev_ptr);

        // fill everything with zeros
        for (size_t i = 0; i < m_elem_num; ++i) {
            m_host_ptr[i] = __float2half(0);
        }

        // fill with csr values
        for (size_t row = 0; row < m_row; row++)
        {
            for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) 
            {
                size_t col = csrColIdx_host[j];
                half val = csrVal_host[j];
                m_host_ptr[row * m_col + col] = val;
            }
        }

        cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice);
    }

    void csrToBcsr() {
        // first prepare the info arrays
        //get the number of row regions
        size_t numColRegions = (m_col + MMA_K - 1) / MMA_K;
        //get the number of column regions
        size_t numRowRegions = (m_row + MMA_M - 1) / MMA_M;
        //get the number of blocks
        numberOfBlocks = numRowRegions * numColRegions;
        //printf("numblocks %d\n", numberOfBlocks);
        //allocate an array of 0's for the blocks
        blockInfo_host = (int *) calloc(sizeof(int), numberOfBlocks);
        // 0 - zero block
        // 1 - 2:4 sparse block
        // 2 - dense block
        //For each of the rows
        //https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL&ig_expand=822,2629,140,92
        __m256i ncolregions = _mm256_set1_epi64x(numColRegions);
        __m256i nrowregions = _mm256_set1_epi64x(numRowRegions);
        __m256i m = _mm256_set1_epi64x(MMA_M);
        __m256i k = _mm256_set1_epi64x(MMA_K);

        size_t* blockindex_nums_16 = (size_t *) malloc(sizeof(size_t)*8);
        size_t* blockindex_nums = (size_t *) calloc(sizeof(size_t), 4);
        for (size_t row = 0; row < m_row; row++)
        {
            
            //for each of the csrRowPtr_host row to row + 1
            //so we can do some sort of SIMD here and then this for any remaining

            //my code starts here
            //lets start with 256 bit SIMD (implement 512 bit SIMD later)
            size_t j = csrRowPtr_host[row];
            __m256i r = _mm256_set1_epi64x(row);
            size_t rdivm = row / MMA_M;
            __m256i b = _mm256_set1_epi64x(rdivm*numColRegions);
            for(; j+8 < csrRowPtr_host[row + 1]; j = j + 8){
                __m256i colRegion1 = _mm256_set_epi64x(csrColIdx_host[j+3]/MMA_K, csrColIdx_host[j+2]/MMA_K, csrColIdx_host[j+1]/MMA_K, csrColIdx_host[j]/MMA_K);
                __m256i colRegion2 = _mm256_set_epi64x(csrColIdx_host[j+7]/MMA_K, csrColIdx_host[j+6]/MMA_K, csrColIdx_host[j+5]/MMA_K, csrColIdx_host[j+4]/MMA_K);
                __m256i blockIndex1 = _mm256_add_epi64(b, colRegion1);
                __m256i blockIndex2 = _mm256_add_epi64(b, colRegion2);
                _mm256_storeu_si256((__m256i*)blockindex_nums_16, blockIndex1);
                _mm256_storeu_si256((__m256i*)&blockindex_nums_16[4], blockIndex2);
                for (size_t i = 0; i < 8; i = i + 1){
                    if (blockInfo_host[blockindex_nums_16[i]] == 0)  // zero block, stops being 0, becomes sparse
                    {
                        //we set it to 1
                        blockInfo_host[blockindex_nums_16[i]] = 1;
                        //add one to nonzero blocks
                        nonzeroBlocks += 1;
                        //increase count of sparse blocks
                        sparseBlocks++;
                    }
                    //same here
                    else if (blockInfo_host[blockindex_nums_16[i]] == 1)  // sparse block
                    {
                        // check can it still be sparse
                        // should I check previous two or I am in new part for 2:4
                        size_t relative24Index = csrColIdx_host[j] % 4;
                        //and here
                        if (relative24Index == 2 || relative24Index == 3)
                        {
                            //check
                            if (j >= csrRowPtr_host[row] + 2 && csrColIdx_host[j-1] == csrColIdx_host[j]-1 && csrColIdx_host[j-2] == csrColIdx_host[j]-2) 
                            {
                                blockInfo_host[blockindex_nums_16[i]] = 2;
                                denseBlocks++;
                                sparseBlocks--;
                            }
                        }
                        
                    }
                }
            }


            for(; j+4 < csrRowPtr_host[row + 1]; j = j + 4){
                //load 8 cols at once
                //size_t colRegion = col / MMA_K;
                __m256i colRegion = _mm256_set_epi64x(csrColIdx_host[j+3]/MMA_K, csrColIdx_host[j+2]/MMA_K, csrColIdx_host[j+1]/MMA_K, csrColIdx_host[j]/MMA_K);

                __m256i blockIndex = _mm256_add_epi64(b, colRegion);

                _mm256_store_si256((__m256i*)blockindex_nums, blockIndex);
                for (size_t i = 0; i < 4; i = i + 1){
                    if (blockInfo_host[blockindex_nums[i]] == 0)  // zero block, stops being 0, becomes sparse
                    {
                        //we set it to 1
                        blockInfo_host[blockindex_nums[i]] = 1;
                        //add one to nonzero blocks
                        nonzeroBlocks += 1;
                        //increase count of sparse blocks
                        sparseBlocks++;
                    }
                    //same here
                    else if (blockInfo_host[blockindex_nums[i]] == 1)  // sparse block
                    {
                        // check can it still be sparse
                        // should I check previous two or I am in new part for 2:4
                        size_t relative24Index = csrColIdx_host[j] % 4;
                        //and here
                        if (relative24Index == 2 || relative24Index == 3)
                        {
                            //check
                            if (j >= csrRowPtr_host[row] + 2 && csrColIdx_host[j-1] == csrColIdx_host[j]-1 && csrColIdx_host[j-2] == csrColIdx_host[j]-2) 
                            {
                                blockInfo_host[blockindex_nums[i]] = 2;
                                denseBlocks++;
                                sparseBlocks--;
                            }
                        }
                        
                    }
                }
                //do predicate calculation (for both the if and esle if)
                //then mask what is occuring
                
                //do checks and then take the relevant action
            }


            for (; j < csrRowPtr_host[row + 1]; j++) 
            {
                //we set col to csrColIdx_host[j]
                size_t col = csrColIdx_host[j];
                //printf("%f\n", csrValA[j]);
                //printf("col %d\n", col);
                //we set the row and col region
                size_t rowRegion = row / MMA_M;
                size_t colRegion = col / MMA_K;
                //printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
                //we get the block index
                size_t blockIndex = rowRegion * numColRegions + colRegion;
                //printf("block  index %d\n", blockIndex);
                //If our block is 0
                //This if statement can be a warp type instruction
                if (blockInfo_host[blockIndex] == 0)  // zero block, stops being 0, becomes sparse
                {
                    //we set it to 1
                    blockInfo_host[blockIndex] = 1;
                    //add one to nonzero blocks
                    nonzeroBlocks += 1;
                    //increase count of sparse blocks
                    sparseBlocks++;
                }
                //same here
                else if (blockInfo_host[blockIndex] == 1)  // sparse block
                {
                    // check can it still be sparse
                    // should I check previous two or I am in new part for 2:4
                    size_t relative24Index = col % 4;
                    //and here
                    if (relative24Index == 2 || relative24Index == 3)
                    {
                        //check
                        if (j >= csrRowPtr_host[row] + 2 && csrColIdx_host[j-1] == col-1 && csrColIdx_host[j-2] == col-2) 
                        {
                            blockInfo_host[blockIndex] = 2;
                            denseBlocks++;
                            sparseBlocks--;
                        }
                    }
                    
                }
            }
        }
        free(blockindex_nums);
        free(blockindex_nums_16);

        //we set relativeIndex to 0
        size_t relativeIndex = 0;
        //we make an array of ints for relativeBlockIndexMapping_host
        relativeBlockIndexMapping_host = (int*) malloc(numberOfBlocks * sizeof(int));
        //This can be SIMD

        for (size_t i = 0; i < numberOfBlocks; i++)
        {
            relativeBlockIndexMapping_host[i] = (blockInfo_host[i] != 0) ? relativeIndex++ : -1;
            //printf("relative [%d] = %d\n", i, relativeBlockIndexMapping[i]);
        }

        // get the bcsr
        bcsrRowPtr_host = (int*)calloc(sizeof(int), (m_row / MMA_M + 1));
        bcsrColIdx_host = (int*)malloc(nonzeroBlocks * sizeof(int));
        bcsrVal_host = (half*)calloc(sizeof(half), nonzeroBlocks * MMA_M * MMA_K);

        size_t num_blocks = 0;
        
        // Do the rowPtrBcsr and colIdxBcsr (SIMD assignment)
        size_t* store_blocks = (size_t*)calloc(sizeof(size_t), 4);
        size_t* store_blocks_16 = (size_t*)calloc(sizeof(size_t), 8);
        size_t jump_16 = MMA_K*8;
        size_t jump = MMA_K*4;
        for (size_t row = 0; row < m_row; row += MMA_M) {
            size_t col = 0;
            // printf("Problem in 314?\n");
            bcsrRowPtr_host[row / MMA_M] = num_blocks; // Update rowPtr
            //printf("rowPtr[%d] = %d\n", row/MMA_M, num_blocks);
            
            __m256i con = _mm256_set1_epi64x(row/MMA_M * numColRegions);
            for (; col + jump_16 < m_col; col += jump_16){
                __m256i coldk1 = _mm256_set_epi64x((col+3*MMA_K)/MMA_K, (col+2*MMA_K)/MMA_K, (col+1*MMA_K)/MMA_K, (col)/MMA_K);
                __m256i coldk2 = _mm256_set_epi64x((col+7*MMA_K)/MMA_K, (col+6*MMA_K)/MMA_K, (col+5*MMA_K)/MMA_K, (col+4*MMA_K)/MMA_K);
                
                __m256i current_block1 = _mm256_add_epi64(con, coldk1);
                __m256i current_block2 = _mm256_add_epi64(con, coldk2);
                
                _mm256_storeu_si256((__m256i*)store_blocks_16, current_block1);
                _mm256_storeu_si256((__m256i*)&store_blocks_16[4], current_block2);
                for (size_t i = 0; i < 8; i = i + 1){
                    int temp = store_blocks_16[i];
                    temp = bcsrColIdx_host[num_blocks];
                    if (blockInfo_host[store_blocks_16[i]] != 0){
                        bcsrColIdx_host[num_blocks] = col + i*MMA_K;
                        num_blocks++;
                    }
                }
            }


            for (; col + jump < m_col; col += jump){
                __m256i coldk = _mm256_set_epi64x((col+3*MMA_K)/MMA_K, (col+2*MMA_K)/MMA_K, (col+1*MMA_K)/MMA_K, (col)/MMA_K);
                __m256i current_block = _mm256_add_epi64(con, coldk);
                _mm256_storeu_si256((__m256i*)store_blocks, current_block);
                for (size_t i = 0; i < 4; i = i + 1){
                    if (blockInfo_host[store_blocks[i]] != 0){
                        bcsrColIdx_host[num_blocks] = col + i*MMA_K;
                        num_blocks++;
                    }
                }
            }
            // Iterate through columns same here if possible
            for (; col < m_col; col += MMA_K) {
                size_t current_block = row / MMA_M * numColRegions + col / MMA_K;
                //printf("Problem in 320?");
                if (blockInfo_host[current_block] == 0)
                {
                    continue;
                }
                //printf("Problem in 325?");
                bcsrColIdx_host[num_blocks] = col; // not relative bcsr columns index / MMA_K if want relative
                //printf("colIdx[%d] = %d\n", num_blocks, col);
                num_blocks++;

            }
        }
        free(store_blocks);
        free(store_blocks_16);

        //printf("Problem in 372?");
        bcsrRowPtr_host[m_row / MMA_M] = num_blocks; // Update last entry of rowPtr
        //printf("rowPtr[%d] = %d\n", numRows / MMA_M, num_blocks);

            
        //printf("%d total blocks\n", totalNumberOfBlocks);
        
        // Do the valuesBcsr
        //This can be calculated as a series of SIMD operations (lets assume 32 bit simd operations)
        size_t* bcsrIndecies = (size_t *)calloc(sizeof(size_t), 4);
        size_t* bcsrIndecies_16 = (size_t *)calloc(sizeof(size_t), 8);
        size_t* blockIndex_vals = (size_t *)calloc(sizeof(size_t), 8);
        for (size_t row = 0; row < m_row; row++)
        {
            size_t j = csrRowPtr_host[row];
            //my code starts here
            //MMA_M*MMA_K
            size_t mbykt = MMA_M*MMA_K;
            __m256i mbyk = _mm256_set1_epi64x(mbykt);
            size_t rdm = row / MMA_M;
            __m256i rowRegion = _mm256_set1_epi64x(rdm);
            size_t rmodmt = row % MMA_M * MMA_K;
            __m256i rmodm = _mm256_set1_epi64x(rmodmt);
            __m256i b = _mm256_set1_epi64x(rdm*numColRegions);
            for(; j+8 < csrRowPtr_host[row + 1]; j = j + 8){
                //load 8 cols at once
                //size_t colRegion = col / MMA_K;
                __m256i colRegion1 = _mm256_set_epi64x(csrColIdx_host[j+3]/MMA_K, csrColIdx_host[j+2]/MMA_K, csrColIdx_host[j+1]/MMA_K, csrColIdx_host[j]/MMA_K);
                __m256i colRegion2 = _mm256_set_epi64x(csrColIdx_host[j+7]/MMA_K, csrColIdx_host[j+6]/MMA_K, csrColIdx_host[j+5]/MMA_K, csrColIdx_host[j+4]/MMA_K);
                //col % MMA_K
                __m256i cmodk1 = _mm256_set_epi64x(csrColIdx_host[j+3]%MMA_K, csrColIdx_host[j+2]%MMA_K, csrColIdx_host[j+1]%MMA_K, csrColIdx_host[j]%MMA_K);
                __m256i cmodk2 = _mm256_set_epi64x(csrColIdx_host[j+7]%MMA_K, csrColIdx_host[j+6]%MMA_K, csrColIdx_host[j+5]%MMA_K, csrColIdx_host[j+4]%MMA_K);
                //size_t offset = row % MMA_M * MMA_K + col % MMA_K;
                __m256i offset1 = _mm256_add_epi64(rmodm, cmodk1);
                __m256i offset2 = _mm256_add_epi64(rmodm, cmodk2);
                //size_t blockIndex = rowRegion * numColRegions + colRegion;
                __m256i blockIndex1 = _mm256_add_epi64(b, colRegion1);
                __m256i blockIndex2 = _mm256_add_epi64(b, colRegion2);
                //relativeBlockIndexMapping_host[blockIndex]
                _mm256_storeu_si256((__m256i*)blockIndex_vals, blockIndex1);
                _mm256_storeu_si256((__m256i*)&blockIndex_vals[4], blockIndex2);
                //relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K
                for(size_t i = 0; i < 8; i++){
                    blockIndex_vals[i] = relativeBlockIndexMapping_host[blockIndex_vals[i]]*mbykt;
                }
                __m256i rbimhibym1 = _mm256_loadu_si256((__m256i*)blockIndex_vals);
                __m256i rbimhibym2 = _mm256_loadu_si256((__m256i*)&blockIndex_vals[4]);
                //size_t bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K + offset;
                __m256i bcsrIndex1 = _mm256_add_epi64(rbimhibym1, offset1);
                __m256i bcsrIndex2 = _mm256_add_epi64(rbimhibym2, offset2);

                _mm256_storeu_si256((__m256i*)bcsrIndecies_16, bcsrIndex1);
                _mm256_storeu_si256((__m256i*)&bcsrIndecies_16[4], bcsrIndex2);
                //As AVX doesnt have scatter and we need to scatter values we do so not as vector instructions
                for(size_t i = 0; i < 8; i++){
                    bcsrVal_host[bcsrIndecies_16[i]] = csrVal_host[j+i];
                }
            }

            for(; j+4 < csrRowPtr_host[row + 1]; j = j + 4){
                //load 8 cols at once
                //size_t colRegion = col / MMA_K;
                __m256i colRegion = _mm256_set_epi64x(csrColIdx_host[j+3]/MMA_K, csrColIdx_host[j+2]/MMA_K, csrColIdx_host[j+1]/MMA_K, csrColIdx_host[j]/MMA_K);
                //col % MMA_K
                __m256i cmodk = _mm256_set_epi64x(csrColIdx_host[j+3]%MMA_K, csrColIdx_host[j+2]%MMA_K, csrColIdx_host[j+1]%MMA_K, csrColIdx_host[j]%MMA_K);
                //size_t offset = row % MMA_M * MMA_K + col % MMA_K;
                __m256i offset = _mm256_add_epi64(rmodm, cmodk);
                //size_t blockIndex = rowRegion * numColRegions + colRegion;
                __m256i blockIndex = _mm256_add_epi64(b, colRegion);
                //relativeBlockIndexMapping_host[blockIndex]
                _mm256_storeu_si256((__m256i*)blockIndex_vals, blockIndex);
                //relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K
                for(size_t i = 0; i < 4; i++){
                    blockIndex_vals[i] = relativeBlockIndexMapping_host[blockIndex_vals[i]]*mbykt;
                }
                __m256i rbimhibym = _mm256_loadu_si256((__m256i*)blockIndex_vals);
                //size_t bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K + offset;
                __m256i bcsrIndex = _mm256_add_epi64(rbimhibym, offset);
                _mm256_storeu_si256((__m256i*)bcsrIndecies, bcsrIndex);
                //As AVX doesnt have scatter and we need to scatter values we do so not as vector instructions
                for(size_t i = 0; i < 4; i++){
                    bcsrVal_host[bcsrIndecies[i]] = csrVal_host[j+i];
                }
            }

            for (; j < csrRowPtr_host[row + 1]; j++) 
            {
                size_t col = csrColIdx_host[j];
                //printf("col %d\n", col);
                size_t rowRegion = row / MMA_M;
                size_t colRegion = col / MMA_K;
                //printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
                size_t blockIndex = rowRegion * numColRegions + colRegion;
                half val = csrVal_host[j];
                //printf("val %f\n", val);
                size_t offset = row % MMA_M * MMA_K + col % MMA_K;
                size_t bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K + offset;
                //printf("relativeIndex %d x %d +  offset %d = %d\n", relativeBlockIndexMapping[blockIndex], blockSize, offset, bcsrIndex);
                bcsrVal_host[bcsrIndex] = val;
            }
        }
        free(bcsrIndecies);
        free(bcsrIndecies_16);
        free(blockIndex_vals);

        /* for (int i = 0; i < num_blocks * blockSize; i++)
        {
            if (i % blockSize == 0)
            {
                printf("\n");   
            }
            printf("%f\n", *(*valuesBcsr + i));
            
        } */

        // create the data structures for locations of nnz blocks
        // not needed for now

    }

    void csrToBcsrKnapsacking() {
        //Find number of row and column regions
        size_t numColRegions = (m_col + MMA_K - 1) / MMA_K;
        size_t numRowRegions = (m_row + MMA_M - 1) / MMA_M;
        //make a vector of the number of 
        std::vector<std::vector<std::tuple<size_t, size_t, float, long, size_t>>> startsEndsOfCols(numRowRegions);

        bcsrRowPtr_host = (int*)calloc(sizeof(int), (m_row / MMA_M + 1));
        std::vector<size_t> vecOfColIdx;
        for (size_t row = 0; row < m_row; row += MMA_M) {
            bcsrRowPtr_host[row / MMA_M] = nonzeroBlocks;
            //printf("[%lu] = %d\n", (unsigned long) row / MMA_M, nonzeroBlocks);
            std::vector<size_t> columnsInBlock;
            for (size_t pointer = row; pointer < row + MMA_M; pointer++) {
                // dodaj iteraciju po columnima
                for (size_t j = csrRowPtr_host[pointer]; j < csrRowPtr_host[pointer+1]; j++) {
                    // columnsInBlock.push_back(csrColIdx_host[j]);
                    startsEndsOfCols[row / MMA_M].push_back(std::make_tuple(csrColIdx_host[j], pointer, __half2float(csrVal_host[j]), -1, 0));
                }
            }
            //std::sort(columnsInBlock.begin(), columnsInBlock.end());
            std::sort(startsEndsOfCols[row / MMA_M].begin(), startsEndsOfCols[row / MMA_M].end(),
            [](const std::tuple<size_t, size_t, float, long, size_t>& tuple1, const std::tuple<size_t, size_t, float, long, size_t>& tuple2) {
                  return std::get<0>(tuple1) < std::get<0>(tuple2);
              });

            for (int i = 0; i < startsEndsOfCols[row / MMA_M].size(); i++) {
                auto a = startsEndsOfCols[row / MMA_M][i];
                //printf("%lu %lu %f %ld %lu\n", std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), std::get<4>(a));
            }

            //if (columnsInBlock.empty()) {
            if (startsEndsOfCols[row / MMA_M].empty()) {
                continue;
            }

            /* size_t start = columnsInBlock[0];
            size_t end = start + MMA_K;
            startsEndsOfCols[row / MMA_M].push_back(start);
            startsEndsOfCols[row / MMA_M].push_back(end);
            nonzeroBlocks++;
            for (size_t i = 1; i < columnsInBlock.size(); ++i) {
                if (columnsInBlock[i] >= end) {
                    start = columnsInBlock[i];
                    end = start + MMA_K;
                    startsEndsOfCols[row / MMA_M].push_back(start);
                    startsEndsOfCols[row / MMA_M].push_back(end);
                    nonzeroBlocks++;
                }
            } */

            size_t firstColumn = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
            size_t lastColumn = std::get<0>(startsEndsOfCols[row / MMA_M].back());
            size_t start;
            size_t span = lastColumn - firstColumn + 1;
            size_t potentialAddLeft = firstColumn - 0;
            size_t potentialAddRight = m_col - lastColumn - 1;
            size_t to_add = MMA_K - span % MMA_K;

            if (span % MMA_K == 0) {
                start = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
            }
            else {
                
                if (potentialAddRight >= to_add) {
                    start = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
                }
                else if (potentialAddLeft >= to_add) {
                    start = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - to_add;
                }
                else {
                    start = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - potentialAddLeft;
                }
            }
             
            vecOfColIdx.push_back(start);
            size_t end = start + MMA_K;
            std::get<3>(startsEndsOfCols[row / MMA_M][0]) = nonzeroBlocks;
            // change relative column
            std::get<4>(startsEndsOfCols[row / MMA_M][0]) = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - start;
            //nonzeroBlocks++;
            /* auto a = startsEndsOfCols[row / MMA_M][0];
            printf("%lu %lu %f %ld    ==>\n", std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
            printf("span %lu addLeft %lu addRight %lu toAdd %lu ", span, potentialAddLeft, potentialAddRight, to_add);
            printf(" start  %lu end %lu \n", start, end); */
            for (size_t i = 1; i < startsEndsOfCols[row / MMA_M].size(); ++i) {
                //a = startsEndsOfCols[row / MMA_M][i];
                //printf("%lu %lu %f %ld    ==>\n", std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a));
                if (std::get<0>(startsEndsOfCols[row / MMA_M][i]) >= end) {
                    // printf("tusam\n");
                    span = lastColumn - std::get<0>(startsEndsOfCols[row / MMA_M][i]) + 1;
                    potentialAddLeft = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - end;
                    to_add = MMA_K - span % MMA_K;
                    //printf("span %lu addLeft %lu addRight %lu toAdd %lu ", span, potentialAddLeft, potentialAddRight, to_add);
                    if (span % MMA_K == 0) {
                        start = std::get<0>(startsEndsOfCols[row / MMA_M][i]);
                    }
                    else {
                        if (potentialAddRight >= to_add) {
                            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]);
                        }
                        else if (potentialAddLeft >= to_add) {
                            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - to_add;
                        }
                        else {
                            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - potentialAddLeft;
                        }
                    }
                    
                    vecOfColIdx.push_back(start);
                    end = start + MMA_K;
                    //printf(" start  %lu end %lu \n", start, end);
                    nonzeroBlocks++;
                }
                
                //printf("cols %lu %lu %lu\n", (unsigned long) start, (unsigned long) end, (unsigned long) nonzeroBlocks);
                std::get<3>(startsEndsOfCols[row / MMA_M][i]) = nonzeroBlocks;
                std::get<4>(startsEndsOfCols[row / MMA_M][i]) = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - start;
            }
            nonzeroBlocks++;
        }
        
        bcsrRowPtr_host[m_row / MMA_M] = nonzeroBlocks;

        //printf ("%lu nnzblocks\n", (unsigned long) nonzeroBlocks);
        bcsrColIdx_host = (int*)malloc(nonzeroBlocks * sizeof(int));
        bcsrVal_host = (half*)calloc(sizeof(half), nonzeroBlocks * MMA_M * MMA_K);

        size_t current_idx = 0;
        for (size_t i = 0; i < nonzeroBlocks; i++) {
            bcsrColIdx_host[i] = vecOfColIdx[i];
            //printf("%d ", bcsrColIdx_host[i]);
        }
        

        //printf("\n");
        for (size_t rowRegion = 0; rowRegion < startsEndsOfCols.size(); rowRegion++) {
            for (size_t element = 0; element < startsEndsOfCols[rowRegion].size(); element++) {
                    size_t col = std::get<0>(startsEndsOfCols[rowRegion][element]);
                    size_t row = std::get<1>(startsEndsOfCols[rowRegion][element]);
                    half val = __float2half(std::get<2>(startsEndsOfCols[rowRegion][element]));
                    size_t relBlock = std::get<3>(startsEndsOfCols[rowRegion][element]);
                    size_t relColumn = std::get<4>(startsEndsOfCols[rowRegion][element]);  
                    bcsrVal_host[relBlock * MMA_M * MMA_K + (row % MMA_M) * MMA_K + relColumn] = val;
                    //printf("%lu %lu %f %ld %lu ==> %lu\n", row, col, __half2float(val), relBlock, relColumn, relBlock * MMA_M * MMA_K + (row % MMA_M) * MMA_K + relColumn);               
                }
                //printf("\n");
        }

        /* printf("\nrowptr\n");
        //printf("%d rowptr size", (int)m_row / MMA_M + 1);
        for (int i = 0; i < m_row / MMA_M + 1; i++) {
            printf("%d ", bcsrRowPtr_host[i]);
        }
        printf("\n");

        for (int i = 0; i < nonzeroBlocks * MMA_M * MMA_K; i++) {
            
            if (i % (MMA_K * MMA_M) == 0) {
                printf ("\n\n");
            }
            else if (i % MMA_K == 0) {
                printf("\n");

            }
            printf("%f ", __half2float(bcsrVal_host[i]));
        } */

        
    }

private:
    size_t m_row = 0;
    size_t m_col = 0;
    std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the hgemm results
    float m_min = -1.0;
    float m_max = 1.0;

    size_t m_elem_num = 0;
    half *m_host_ptr = nullptr;
    half *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(SparseMatrix);

    char* filename = "";

    size_t nnz = 0;

    half *csrVal_host = nullptr;
    int *csrRowPtr_host = nullptr;
    int *csrColIdx_host = nullptr;

    half *bcsrVal_host = nullptr;
    int *bcsrRowPtr_host = nullptr;
    int *bcsrColIdx_host = nullptr; 


    half *csrVal_dev = nullptr;
    int *csrRowPtr_dev = nullptr;
    int *csrColIdx_dev = nullptr;

    half *bcsrVal_dev = nullptr;
    int *bcsrRowPtr_dev = nullptr;
    int *bcsrColIdx_dev = nullptr;

    int *blockInfo_host = nullptr;
    int *relativeBlockIndexMapping_host = nullptr;

    int *blockInfo_dev = nullptr;
    int *relativeBlockIndexMapping_dev = nullptr;

    size_t numberOfBlocks = 0;
    size_t nonzeroBlocks = 0;
    size_t sparseBlocks = 0;
    size_t denseBlocks = 0;
};
