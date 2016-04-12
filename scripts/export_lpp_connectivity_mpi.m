batch_size=512
%batch_size=str2double(getenv('BATCH_SIZE'))

for forest = 1:1000

    lpp_input_file=sprintf('LPPtoDGC.%d.dat',forest)

    tic;
    m = dlmread(lpp_input_file, SEP=' ', R0=1, C0=0);
    toc;

    for batch_index = 1:batch_size
        append_connmatrix('LPPtoDGC',double(m),1000000,batch_index,batch_size,sprintf('B%d',batch_size));
    end

end
