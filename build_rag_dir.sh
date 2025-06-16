# Set up the RAG database

# Make a directory to store the files
mkdir rag_data
cd rag_data

# Download the Vitis HLS User Guide
wget https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_2/ug1399-vitis-hls.pdf

# Clone the Vitis_Libraries repository
git clone git@github.com:Xilinx/Vitis_Libraries.git
# Copy the image processing HLS kernels to the database
cp Vitis_Libraries/vision/L1/include/imgproc/* .
# Remove the Vitis_Libraries database
rm -rf Vitis_Libraries

cd ..
