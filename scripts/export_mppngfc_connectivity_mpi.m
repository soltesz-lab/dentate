batch_size=2048
%batch_size=str2double(getenv('BATCH_SIZE'))
srcbatch_size = 64

for i = 1:srcbatch_size

    mpp_input_file=sprintf('MPPtoNGFC.%d.dat',i-1)

    tic;
    m = dlmread(mpp_input_file, SEP=' ', R0=0, C0=0);
    toc;

    for batch_index = 1:batch_size
        append_connmatrix('MPPtoNGFC',double(m),5000,1044650,batch_index,batch_size,sprintf('B%d',batch_size));
    end

end
