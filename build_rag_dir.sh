# Set up the RAG database

# Make a directory to store the files
mkdir rag_data
cd rag_data

# Download the Vitis HLS User Guide
wget https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_2/ug1399-vitis-hls.pdf

# Clone the Vitis_Libraries repository
git clone https://github.com/Xilinx/Vitis_Libraries.git
# Copy the vision/imgproc HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/imgproc/* .
# Copy the vision/video HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/video/* .
# Copy the vision/features HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/features/* .
# Copy the vision/dnn HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/dnn/* .
# Copy the vision/core HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/core/* .
# Copy the vision/common HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/common/* .

# Copy the blas/helpers/utils HLS kernels to the database
cp Vitis_Libraries/blas/L1/include/hw/xf_blas/helpers/utils/* .
# Copy the blas/helpers/funcs HLS kernels to the database
cp Vitis_Libraries/blas/L1/include/hw/xf_blas/helpers/funcs/* .
# Copy the blas/helpers/dataMover HLS kernels to the database
cp Vitis_Libraries/blas/L1/include/hw/xf_blas/helpers/dataMover/* .
# Copy the blas/gemm HLS kernels to the database
cp Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm/* .
# Copy the blas HLS kernels to the database
cp Vitis_Libraries/blas/L1/include/hw/xf_blas/* .

# Remove the Vitis_Libraries database
rm -rf Vitis_Libraries

cd ..
