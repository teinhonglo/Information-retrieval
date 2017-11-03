read -p "Please input your file name : " filename
read -p "Please input GPU : " gpu
THEANO_FLAGS=device=${gpu}, floatX=float32 python ${filename}






