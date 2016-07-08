batch_size=2048
%batch_size=str2double(getenv('BATCH_SIZE'))

for forest = 1:1000

    mpp_input_file=sprintf('MPPtoDGC.%d.dat',forest)

    tic;
    m = dlmread(mpp_input_file, SEP=' ', R0=1, C0=0);
    toc;

    for batch_index = 1:batch_size
        append_connmatrix('MPPtoDGC',double(m),1000000,0,batch_index,batch_size,sprintf('B%d',batch_size));
    end

end
