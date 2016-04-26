batch_size=512
%batch_size=str2double(getenv('BATCH_SIZE'))
srcbatch_size=48

for i = 1:srcbatch_size

    lpp_input_file=sprintf('LPPtoBC.%d.dat',i)

    tic;
    m = dlmread(lpp_input_file, SEP=' ', R0=1, C0=0);
    toc;

    for batch_index = 1:batch_size
        append_connmatrix('LPPtoBC',double(m),3800,1039001,batch_index,batch_size,sprintf('B%d',batch_size));
    end

end
