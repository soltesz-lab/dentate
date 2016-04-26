batch_size=512
%batch_size=str2double(getenv('BATCH_SIZE'))

for i = 1:batch_size

    mpp_input_file=sprintf('MPPtoBC.%d.dat',i)

    tic;
    m = dlmread(mpp_input_file, SEP=' ', R0=1, C0=0);
    toc;

    for batch_index = 1:batch_size
        append_connmatrix('MPPtoBC',double(m),3800,1039001,batch_index,batch_size,sprintf('B%d',batch_size));
    end

end
