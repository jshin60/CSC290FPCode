# CSC290FPCode
The code in part 1 and 2 is based on the cuda_hgemm code for SMaT which can be found at https://github.com/spcl/smat (the synthetic SpMM test cases can be found from this code repository as well).

Compilation can be done from the part1 or part2 folder by running compile.sh (see the SMaT repository at https://github.com/spcl/smat on exact running details).

The dgl-SpMM data was gathered from https://github.com/OnixHoque/sptransx-mlsys2025-reproduce/ and an example can be found in the example_SpTransX_data_gather folder (transh) the primary changes are that the sparse A matrix is saved as a .mtx file using scipy csr array and mmwrite and the dense B matrix is saved as a .txt file using numpy savetxt. Furthermore, time data was gathered for dgl-SpMM operations.

Any usage of outside guides or documentation is often times linked around where the code was added (as well as inline cited within the report itself when explaining implementation details or how tests were set up). However below are the exact guides, github repositories, and documentation references for any code written for this project (they are also cited within the written report):

Artifact evaluation reproduction for "sparsetransx: Efficient training of translation-based knowledge graph embeddings using sparsematrix operations", mlsys 2025. URL: https://github.com/OnixHoque/sptransx-mlsys2025-reproduce/.

csr_array - scipy v1.16.1 manual. URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array.

Intel® intrinsics guide. URL: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL.

mmwrite - scipy v1.16.1 manual. URL: https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.io.mmwrite.html.

numpy.savetxt - numpy v2.3 manual. URL: https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html.

Smat: (s)parse (ma)trix matrix (t)ensor core-accelerated library. URL:https://github.com/spcl/smat.

GeeksforGeeks. C program to read content of a file, Jul2025. URL: https://www.geeksforgeeks.org/c/c-program-to-read-contents-of-whole-file/.

GeeksforGeeks. C program to read content of a file, Jul2025. URL: https://www.geeksforgeeks.org/c/c-program-to-read-contents-of-whole-file/.

GeeksforGeeks. Removing trailing newline character from fgets() input, Aug 2025. URL:https://www.geeksforgeeks.org/dsa/removing-trailing-newline-character-from-fgets-input/.

W3Schools. C write to files. URL: https://www.w3schools.com/c/c_files_write.php.

W3Schools. Python file write. URL: https://www.w3schools.com/python/python_file_write.asp.
