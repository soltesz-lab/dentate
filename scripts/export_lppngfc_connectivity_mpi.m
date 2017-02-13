batch_size=2048
%batch_size=str2double(getenv('BATCH_SIZE'))
srcbatch_size=64

for i = 1:srcbatch_size

    lpp_input_file=sprintf('LPPtoNGFC.%d.dat',i-1)

    tic;
    m = dlmread(lpp_input_file, SEP=' ', R0=0, C0=0);
    toc;

    for batch_index = 1:batch_size
        append_connmatrix('LPPtoNGFC',double(m),5000,1044650,batch_index,batch_size,sprintf('B%d',batch_size));
    end

end
